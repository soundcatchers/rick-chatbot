#!/usr/bin/env python3
"""
Rick Sanchez Chatbot - Enhanced Edition with Internet Search
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
            'search.yahoo.com'
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
    
    def search_google_scrape(self, query, max_results=3):
        """Search Google using web scraping (improved method)"""
        try:
            # Encode the query for URL
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}&num=10&hl=en"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                content = response.text
                snippets = []
                
                # Multiple patterns to catch different Google layouts
                patterns = [
                    r'<div class="BNeawe[^>]*>([^<]+)</div>',
                    r'<span class="aCOpRe">([^<]+)</span>',
                    r'<div class="VwiC3b[^>]*>([^<]+)</div>',
                    r'<span class="st">([^<]+)</span>',
                    r'<div class="s">([^<]+)</div>',
                    r'<div data-content-feature="1"[^>]*>([^<]+)</div>',
                    r'<div class="IsZvec">([^<]+)</div>',
                    r'<div class="hgKElc">([^<]+)</div>'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        clean_text = re.sub(r'<[^>]+>', '', match).strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        if clean_text and len(clean_text) > 30 and len(clean_text) < 200:
                            snippets.append(clean_text)
                    
                    if len(snippets) >= max_results:
                        break
                
                # If no snippets found, try to get any meaningful text
                if not snippets:
                    # Look for any text that might be a result
                    all_text = re.findall(r'>([^<]{30,150})<', content)
                    for text in all_text:
                        clean_text = text.strip()
                        if (clean_text and 
                            not clean_text.startswith('http') and 
                            not clean_text.startswith('www') and
                            'google' not in clean_text.lower() and
                            len(clean_text) > 30):
                            snippets.append(clean_text)
                            if len(snippets) >= max_results:
                                break
                
                return snippets[:max_results] if snippets else [f"Found search results for '{query}' but couldn't extract clear text"]
                
        except Exception as e:
            return [f"Google search error: {str(e)}"]
    
    def search_simple_api(self, query):
        """Simple search using a free API service"""
        try:
            # Use a free search API (like SerpApi alternative)
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Get abstract if available
                if data.get('Abstract'):
                    results.append(data['Abstract'])
                
                # Get related topics
                if data.get('RelatedTopics'):
                    for topic in data['RelatedTopics'][:2]:
                        if isinstance(topic, dict) and 'Text' in topic:
                            results.append(topic['Text'])
                
                # Get answer if available
                if data.get('Answer'):
                    results.insert(0, data['Answer'])
                
                return results if results else ["No results from backup search"]
                
        except Exception as e:
            return [f"Backup search error: {str(e)}"]
    
    def search_bing_scrape(self, query, max_results=3):
        """Search Bing using web scraping (improved method)"""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.bing.com/search?q={encoded_query}&count=10"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                content = response.text
                snippets = []
                
                # Multiple patterns for Bing
                patterns = [
                    r'<p class="b_paractl">([^<]+)</p>',
                    r'<div class="b_caption">.*?<p>([^<]+)</p>',
                    r'<div class="b_snippetBigText">([^<]+)</div>',
                    r'<span class="b_snippetText">([^<]+)</span>'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        clean_text = re.sub(r'<[^>]+>', '', match).strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text)
                        if clean_text and len(clean_text) > 30 and len(clean_text) < 200:
                            snippets.append(clean_text)
                    
                    if len(snippets) >= max_results:
                        break
                
                return snippets[:max_results] if snippets else [f"Found Bing results for '{query}' but couldn't extract clear text"]
                
        except Exception as e:
            return [f"Bing search error: {str(e)}"]
    
    def search_web(self, query):
        """Search the web using available methods"""
        if not self.has_internet:
            return None
            
        print(f"{Fore.YELLOW}*burp* Searching for: {query}...{Style.RESET_ALL}")
        
        # Try Google first
        try:
            results = self.search_google_scrape(query)
            if results and len(results) > 0 and "error" not in results[0].lower():
                print(f"{Fore.GREEN}*burp* Found some interdimensional knowledge!{Style.RESET_ALL}")
                return results
        except Exception as e:
            print(f"{Fore.RED}*burp* Google search failed: {str(e)}{Style.RESET_ALL}")
            
        # Fallback to Bing
        try:
            results = self.search_bing_scrape(query)
            if results and len(results) > 0 and "error" not in results[0].lower():
                print(f"{Fore.GREEN}*burp* Found backup results from Bing!{Style.RESET_ALL}")
                return results
        except Exception as e:
            print(f"{Fore.RED}*burp* Bing search also failed: {str(e)}{Style.RESET_ALL}")
            
        # Try a simple API-based search as last resort
        try:
            return self.search_simple_api(query)
        except:
            pass
            
        return [f"*burp* Sorry, all search methods failed for '{query}'. The internet is being difficult today."]

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
        'breaking', 'just announced', 'new release', 'recently'
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
    print(f"{Fore.CYAN}🧪 Rick Sanchez Chatbot - Enhanced Edition with Internet Search{Style.RESET_ALL}")
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
                    print(f"{Fore.YELLOW}*burp* Searching the interdimensional internet...{Style.RESET_ALL}")
                    search_results = search_engine.search_web(search_query)
                    user_input = f"Tell me about: {search_query}"
                else:
                    print(f"{Fore.RED}Rick: *burp* Listen, genius, we don't have internet! I can't search for squat!{Style.RESET_ALL}")
                    continue
            
            # Check if we should search automatically
            elif internet_available and needs_search(user_input):
                print(f"{Fore.YELLOW}*burp* This might need fresh info, let me check...{Style.RESET_ALL}")
                search_results = search_engine.search_web(user_input)
                
                # Debug: show what we found
                if search_results:
                    print(f"{Fore.CYAN}Debug - Search results found: {len(search_results)} items{Style.RESET_ALL}")
                    for i, result in enumerate(search_results[:2]):  # Show first 2
                        print(f"{Fore.CYAN}Result {i+1}: {result[:100]}...{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Debug - No search results returned{Style.RESET_ALL}")
                
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
