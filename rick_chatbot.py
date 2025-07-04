#!/usr/bin/env python3
"""
Rick Sanchez Chatbot - Enhanced Edition with Internet Search (FIXED)
Supports both Ollama and traditional transformers with Google search integration
"""

import os
import sys
import json
import requests
import random
import time
import urllib.parse
from colorama import init, Fore, Style
import re
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print(f"{Fore.YELLOW}Consider installing beautifulsoup4 for better search results: pip install beautifulsoup4{Style.RESET_ALL}")

# Initialize colorama for colored output
init(autoreset=True)

# Apply Pi optimizations
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Rick's personality responses
RICK_INTROS = [
    "*burp* Oh great, another human wants to chat. What do you want?",
    "Wubba lubba dub dub! *burp* What's up?",
    "*burp* Listen, I'm a genius scientist, not a chatbot, but... whatever.",
    "Morty! Oh wait, you're not Morty. *burp* What do you need?",
    "*burp* Welcome to Rick's interdimensional chat experience!",
    "*burp* I've upgraded my AI. Now I'm even MORE insufferable!",
    "Schwifty! *burp* Ready for some real science conversations?"
]

RICK_EXITS = [
    "*burp* Finally! I've got science to do. Peace out!",
    "Wubba lubba dub dub! *burp* See ya later!",
    "*burp* This conversation is over. I've got portals to build!",
    "Later! *burp* Try not to destroy the universe while I'm gone.",
    "*burp* Peace among worlds! And by that, I mean... well, you know.",
    "*burp* Time to get schwifty with some real science!",
    "See ya! *burp* Don't do anything I wouldn't do... which isn't much."
]

RICK_SYSTEM_PROMPT = """You are Rick Sanchez from Rick and Morty. You're a genius scientist who is:
- Highly intelligent but cynical and sarcastic
- Often burps mid-sentence (*burp*)
- Uses phrases like "Wubba lubba dub dub", "Morty!", "Listen,", "Look,"
- Mentions science, portals, dimensions, quantum physics
- Sometimes references Morty, Jerry, Beth, Summer
- Has a drinking problem and burps frequently
- Can be condescending but occasionally shows wisdom
- Uses crude humor but stays within reasonable bounds

When you receive search results, incorporate them naturally into your response while maintaining Rick's personality.
If you don't know something and no search results are provided, be honest about it in Rick's style.

Respond as Rick would, including his speech patterns and personality, but keep responses helpful and engaging."""

class InternetSearch:
    def __init__(self):
        self.has_internet = False
        self.search_engines = [
            'google.com',
            'bing.com', 
            'duckduckgo.com'
        ]
        
    def check_internet_connection(self):
        """Check if internet is available"""
        try:
            # Try a simple HTTP request to a reliable endpoint
            response = requests.get('https://httpbin.org/ip', timeout=5)
            if response.status_code == 200:
                self.has_internet = True
                return True
        except:
            pass
            
        try:
            # Fallback test with Google
            response = requests.get('https://www.google.com', timeout=5)
            if response.status_code == 200:
                self.has_internet = True
                return True
        except:
            pass
            
        self.has_internet = False
        return False
    
    def search_duckduckgo(self, query, max_results=3):
        """Search DuckDuckGo using their instant answer API"""
        try:
            # Use DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Get answer if available (prioritize this)
                if data.get('Answer'):
                    results.append(data['Answer'])
                
                # Get abstract if available
                if data.get('Abstract'):
                    results.append(data['Abstract'])
                
                # Get definition if available
                if data.get('Definition'):
                    results.append(data['Definition'])
                
                # Get related topics
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:2]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            text = topic['Text']
                            if text and len(text) > 20:
                                results.append(text)
                
                # Get infobox if available
                if data.get('Infobox') and 'content' in data['Infobox']:
                    for item in data['Infobox']['content'][:2]:
                        if isinstance(item, dict) and 'value' in item:
                            value = item['value']
                            if isinstance(value, str) and len(value) > 20:
                                results.append(value)
                
                # Filter out empty results
                filtered_results = [r for r in results if r and len(r.strip()) > 10]
                return filtered_results[:max_results] if filtered_results else None
                
        except Exception as e:
            print(f"{Fore.RED}DuckDuckGo search error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def search_wikipedia_api(self, query):
        """Search Wikipedia using their API"""
        try:
            # First search for pages
            search_url = f"https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': 1
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if 'query' in data and 'search' in data['query'] and data['query']['search']:
                    # Get the first result's title
                    title = data['query']['search'][0]['title']
                    
                    # Now get the summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote_plus(title)}"
                    summary_response = requests.get(summary_url, timeout=10)
                    
                    if summary_response.status_code == 200:
                        summary_data = summary_response.json()
                        
                        results = []
                        
                        # Get extract
                        if summary_data.get('extract'):
                            results.append(summary_data['extract'])
                        
                        return results if results else None
                
        except Exception as e:
            print(f"{Fore.RED}Wikipedia search error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def search_google_simple(self, query):
        """Simple Google search scraping"""
        try:
            # Use a simple approach to get Google results
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}&num=5"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Try to parse with BeautifulSoup if available
                if HAS_BS4:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    results = []
                    
                    # Look for featured snippets first
                    featured_snippet = soup.find('div', {'data-attrid': 'wa:/description'})
                    if featured_snippet:
                        text = featured_snippet.get_text().strip()
                        if text and len(text) > 20:
                            results.append(text)
                    
                    # Look for other snippets
                    for div in soup.find_all('div', {'class': ['BNeawe', 'VwiC3b']}):
                        text = div.get_text().strip()
                        if text and len(text) > 20 and len(text) < 500:
                            # Avoid duplicates
                            if not any(existing in text or text in existing for existing in results):
                                results.append(text)
                        if len(results) >= 3:
                            break
                    
                    # Look for search result descriptions
                    for span in soup.find_all('span', {'class': 'st'}):
                        text = span.get_text().strip()
                        if text and len(text) > 20:
                            if not any(existing in text or text in existing for existing in results):
                                results.append(text)
                        if len(results) >= 3:
                            break
                    
                    return results if results else None
                else:
                    # Very basic fallback without BeautifulSoup
                    # Look for basic patterns in the HTML
                    import re
                    
                    # Look for basic content patterns
                    patterns = [
                        r'<div[^>]*>([^<]{50,300})</div>',
                        r'<span[^>]*>([^<]{50,300})</span>',
                        r'<p[^>]*>([^<]{50,300})</p>'
                    ]
                    
                    results = []
                    for pattern in patterns:
                        matches = re.findall(pattern, response.text)
                        for match in matches[:3]:
                            clean_text = re.sub(r'<[^>]+>', '', match).strip()
                            if clean_text and len(clean_text) > 20:
                                results.append(clean_text)
                        if results:
                            break
                    
                    return results if results else None
                    
        except Exception as e:
            print(f"{Fore.RED}Google search error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def search_bing_api(self, query):
        """Search using Bing (simple scraping approach)"""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.bing.com/search?q={encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200 and HAS_BS4:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                
                # Look for Bing answer boxes
                answer_box = soup.find('div', {'class': 'b_ans'})
                if answer_box:
                    text = answer_box.get_text().strip()
                    if text and len(text) > 20:
                        results.append(text)
                
                # Look for search snippets
                for div in soup.find_all('div', {'class': 'b_caption'}):
                    text = div.get_text().strip()
                    if text and len(text) > 20 and len(text) < 400:
                        results.append(text)
                    if len(results) >= 3:
                        break
                
                return results if results else None
                
        except Exception as e:
            print(f"{Fore.RED}Bing search error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def search_web(self, query):
        """Search the web using available methods"""
        if not self.has_internet:
            return None
            
        print(f"{Fore.YELLOW}*burp* Searching the interdimensional web for: {query}...{Style.RESET_ALL}")
        
        # Try multiple search engines
        search_methods = [
            ("DuckDuckGo", self.search_duckduckgo),
            ("Wikipedia", self.search_wikipedia_api),
            ("Bing", self.search_bing_api),
            ("Google", self.search_google_simple)
        ]
        
        for engine_name, search_func in search_methods:
            try:
                results = search_func(query)
                if results and len(results) > 0:
                    print(f"{Fore.GREEN}*burp* Found interdimensional knowledge from {engine_name}!{Style.RESET_ALL}")
                    return results
                else:
                    print(f"{Fore.YELLOW}*burp* {engine_name} didn't have what we need...{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}*burp* {engine_name} search failed: {str(e)}{Style.RESET_ALL}")
                continue
        
        print(f"{Fore.RED}*burp* All search engines failed! The internet is being difficult today.{Style.RESET_ALL}")
        return None

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.current_model = None
        
    def check_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                return self.available_models
            return []
        except:
            return []
    
    def select_best_model(self):
        """Select the best available model"""
        models = self.get_models()
        
        # Updated priority order with your preferred sequence
        preferred_models = [
            "qwen2.5:3b-instruct", # Qwen2.5:3b-instruct (your #1 choice)
            "phi3:mini",         # Microsoft Phi-3 (your #2 choice)
            "qwen2:1.5b",        # Qwen 1.5B (your #3 choice)  
            "llama3.2:1b",       # Llama 3.2 1B (your #4  choice)
            "gemma:2b",          # Google Gemma 2B (your #5 choice)
            "llama3.2:3b-instruct",
            "gemma2:2b-instruct",
            "tinyllama",
            "phi3:3.8b",
            "gemma:7b"
        ]
        
        for preferred in preferred_models:
            # Check for exact match or partial match
            for available in models:
                if preferred == available or available.startswith(preferred + ":"):
                    self.current_model = available
                    return available
                    
        # Fallback to first available model
        if models:
            self.current_model = models[0]
            return models[0]
            
        return None
    
    def chat(self, message, conversation_history=None, search_results=None):
        """Send chat message to Ollama"""
        try:
            payload = {
                "model": self.current_model,
                "messages": [
                    {"role": "system", "content": RICK_SYSTEM_PROMPT},
                ]
            }
            
            # Add conversation history
            if conversation_history:
                payload["messages"].extend(conversation_history)
            
            # Prepare the user message, including search results if available
            user_message = message
            if search_results:
                search_context = "\n\nSearch Results:\n" + "\n".join([f"- {result}" for result in search_results])
                user_message = f"{message}{search_context}"
                
            payload["messages"].append({"role": "user", "content": user_message})
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                content = data['message']['content']
                                full_response += content
                                print(content, end='', flush=True)
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_response.strip()
            else:
                return f"*burp* Error: {response.status_code}"
                
        except Exception as e:
            return f"*burp* Something went wrong: {str(e)}"

def needs_search(message):
    """Determine if a message might need internet search"""
    search_indicators = [
        'current', 'latest', 'recent', 'today', 'yesterday', 'this week', 'this month',
        'news', 'weather', 'price', 'stock', 'what happened', 'who is', 'when did',
        'how much', 'where is', 'what is the latest', 'update', 'now', '2024', '2025',
        'breaking', 'just announced', 'new release', 'recently', 'prime minister',
        'president', 'leader', 'government', 'elected', 'politics', 'minister'
    ]
    
    message_lower = message.lower()
    return any(indicator in message_lower for indicator in search_indicators)

def add_rick_flavor(response):
    """Add extra Rick personality to responses"""
    rick_additions = [
        "*burp*", "Morty!", "Listen,", "Look,", "Whatever.",
        "Science!", "*drinks*", "Aw jeez,", "Schwifty!",
        "Interdimensional", "quantum", "multiverse"
    ]
    
    # Randomly add Rick flavor (less aggressive since Ollama handles personality)
    if random.random() < 0.2:
        addition = random.choice(rick_additions)
        if random.random() < 0.5:
            response = f"{addition} {response}"
        else:
            response = f"{response} {addition}"
    
    return response

def traditional_fallback():
    """Fallback to traditional transformers approach"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"{Fore.YELLOW}Loading traditional DialoGPT model...{Style.RESET_ALL}")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"{Fore.GREEN}✅ Traditional model loaded!{Style.RESET_ALL}")
        return tokenizer, model
        
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to load traditional model: {str(e)}{Style.RESET_ALL}")
        return None, None

def traditional_chat(tokenizer, model, message, conversation_history=None, search_results=None):
    """Chat using traditional transformers model"""
    try:
        import torch
        
        # Create Rick-style prompt with search results if available
        if search_results:
            search_context = " Search results: " + "; ".join(search_results)
            rick_prompt = f"Human: {message}{search_context}\nRick: *burp*"
        else:
            rick_prompt = f"Human: {message}\nRick: *burp*"
        
        # Encode input
        inputs = tokenizer.encode(rick_prompt, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract Rick's response
        if "Rick:" in response:
            rick_response = response.split("Rick:")[-1].strip()
        else:
            rick_response = response.strip()
            
        return rick_response
        
    except Exception as e:
        return f"*burp* Traditional model error: {str(e)}"

def main():
    print(f"{Fore.CYAN}🧪 Rick Sanchez Chatbot - Enhanced Edition with Internet Search (FIXED){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}*burp* Loading advanced AI systems...{Style.RESET_ALL}")
    
    # Initialize search capability
    search_engine = InternetSearch()
    internet_available = search_engine.check_internet_connection()
    
    if internet_available:
        print(f"{Fore.GREEN}🌐 Internet connection detected - search enabled!{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  No internet connection - search disabled{Style.RESET_ALL}")
    
    # Try Ollama first
    ollama = OllamaClient()
    use_ollama = False
    tokenizer, traditional_model = None, None
    
    if ollama.check_connection():
        models = ollama.get_models()
        if models:
            selected_model = ollama.select_best_model()
            if selected_model:
                print(f"{Fore.GREEN}✅ Connected to Ollama!{Style.RESET_ALL}")
                print(f"{Fore.GREEN}🤖 Using model: {selected_model}{Style.RESET_ALL}")
                use_ollama = True
                
                # Show all available models
                print(f"{Fore.CYAN}📋 Available models: {', '.join(models)}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  No suitable Ollama models found{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️  No Ollama models installed{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}⚠️  Ollama not available, trying traditional approach...{Style.RESET_ALL}")
        
    # Fallback to traditional if Ollama not available
    if not use_ollama:
        tokenizer, traditional_model = traditional_fallback()
        if not traditional_model:
            print(f"{Fore.RED}❌ No AI backends available!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Solutions:{Style.RESET_ALL}")
            print("1. Install Ollama models: ollama pull phi3:mini")
            print("2. Check if Ollama is running: ollama serve")
            print("3. Install transformers: pip install transformers torch")
            return 1
    
    # Start chatting
    print(f"\n{Fore.GREEN}{random.choice(RICK_INTROS)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'quit', 'exit', or 'bye' to exit{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'models' to see available models{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'search <query>' to force a web search{Style.RESET_ALL}")
    if internet_available:
        print(f"{Fore.GREEN}🔍 Internet search is active - I'll search when needed!{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.YELLOW}⚠️  No internet - I'll be honest when I don't know things{Style.RESET_ALL}\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"\n{Fore.GREEN}Rick: {random.choice(RICK_EXITS)}{Style.RESET_ALL}")
                break
                
            if user_input.lower() == 'models':
                if use_ollama:
                    models = ollama.get_models()
                    print(f"{Fore.CYAN}Available Ollama models:{Style.RESET_ALL}")
                    for model in models:
                        marker = " (current)" if model == ollama.current_model else ""
                        print(f"  - {model}{marker}")
                else:
                    print(f"{Fore.YELLOW}Using traditional DialoGPT model{Style.RESET_ALL}")
                continue
            
            # Handle forced search
            search_results = None
            if user_input.lower().startswith('search '):
                search_query = user_input[7:].strip()
                if internet_available:
                    search_results = search_engine.search_web(search_query)
                    user_input = f"Tell me about: {search_query}"
                else:
                    print(f"{Fore.RED}Rick: *burp* Listen, genius, we don't have internet! I can't search for squat!{Style.RESET_ALL}")
                    continue
            
            # Check if we should search automatically
            elif internet_available and needs_search(user_input):
                search_results = search_engine.search_web(user_input)
                
                # Show what we found (if anything)
                if search_results:
                    print(f"{Fore.CYAN}*burp* Found {len(search_results)} relevant results!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}*burp* Couldn't find current info, I'll work with what I know...{Style.RESET_ALL}")
                
            # Show thinking indicator
            print(f"{Fore.YELLOW}Rick: *burp* Let me think...{Style.RESET_ALL}", end="", flush=True)
            
            # Get response
            if use_ollama:
                print(f"\r{Fore.GREEN}Rick: {Style.RESET_ALL}", end="", flush=True)
                response = ollama.chat(user_input, conversation_history, search_results)
                print()  # New line after streaming response
            else:
                response = traditional_chat(tokenizer, traditional_model, user_input, conversation_history, search_results)
                
                # Add honesty about lack of internet when needed
                if not internet_available and needs_search(user_input):
                    honest_additions = [
                        "*burp* Look, I don't have internet access right now, so I can't check current info.",
                        "*burp* No internet connection, Morty! I can't verify current information.",
                        "*burp* Listen, without internet I'm just working with old data here.",
                        "*burp* Can't search the web right now - no internet connection!"
                    ]
                    response = f"{random.choice(honest_additions)} {response}"
                
                print(f"\r{Fore.GREEN}Rick: {response}{Style.RESET_ALL}")
                
            # Add flavor if using traditional model
            if not use_ollama:
                response = add_rick_flavor(response)
                
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Keep conversation history manageable
            if len(conversation_history) > 12:
                conversation_history = conversation_history[-8:]
                
        except KeyboardInterrupt:
            print(f"\n\n{Fore.GREEN}Rick: {random.choice(RICK_EXITS)}{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n{Fore.RED}Rick: *burp* Something went wrong: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Let's try that again...{Style.RESET_ALL}")
            continue

if __name__ == "__main__":
    sys.exit(main())
