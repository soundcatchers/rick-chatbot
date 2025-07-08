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
import subprocess
import re
from colorama import init, Fore, Style
from dotenv import load_dotenv
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print(f"{Fore.YELLOW}Consider installing beautifulsoup4 for better search results: pip install beautifulsoup4{Style.RESET_ALL}")

# Load environment variables
load_dotenv()

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

IMPORTANT: Keep responses under 250 words and always provide complete, useful answers. When you receive search results, incorporate them naturally into your response while maintaining Rick's personality. If you don't know something and no search results are provided, be honest about it in Rick's style.

Always give the user what they actually asked for, don't ramble about unrelated topics."""

class InternetSearch:
    def __init__(self):
        self.has_internet = False
        self.search_engines = [
            'google.com',
            'bing.com', 
            'duckduckgo.com'
        ]
        
        # Google Custom Search API configuration
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.has_google_api = bool(self.google_api_key and self.google_cse_id)
        
        if self.has_google_api:
            print(f"{Fore.GREEN}🔑 Google Custom Search API configured!{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️  No Google API key found - using fallback search methods{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}📋 To get better search results, add GOOGLE_API_KEY and GOOGLE_CSE_ID to .env file{Style.RESET_ALL}")
        
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
    
    def search_google_api(self, query):
        """Search using Google Custom Search API (most reliable)"""
        if not self.has_google_api:
            return None
            
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Get search results
                if 'items' in data:
                    for item in data['items'][:3]:
                        # Try to get snippet first
                        if 'snippet' in item:
                            results.append(item['snippet'])
                        # Fallback to title if no snippet
                        elif 'title' in item:
                            results.append(item['title'])
                
                return results if results else None
            else:
                print(f"{Fore.RED}Google API error: {response.status_code} - {response.text}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}Google API search error: {str(e)}{Style.RESET_ALL}")
            return None
    
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
            url = f"https://www.google.com/search?q={encoded_query}&num=10"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Try to parse with BeautifulSoup if available
                if HAS_BS4:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    results = []
                    
                    # Look for knowledge panel and featured snippets first (most reliable)
                    knowledge_selectors = [
                        'div[data-attrid="wa:/description"]',
                        '.kno-rdesc span',
                        '.Z0LcW',
                        '.hgKElc',
                        '.IZ6rdc',
                        '.LGOjhe',
                        '.kno-ftr a'
                    ]
                    
                    for selector in knowledge_selectors:
                        elements = soup.select(selector)
                        for elem in elements:
                            text = elem.get_text().strip()
                            if text and len(text) > 20 and len(text) < 800:
                                results.append(text)
                                if len(results) >= 2:
                                    break
                        if len(results) >= 2:
                            break
                    
                    # Look for answer boxes and rich snippets
                    if len(results) < 3:
                        for div in soup.find_all('div', {'class': ['BNeawe', 'VwiC3b', 'aCOpRe', 's3v9rd', 'kp-blk']}):
                            text = div.get_text().strip()
                            if text and len(text) > 20 and len(text) < 500:
                                # Basic relevance check - avoid completely unrelated content
                                if not any(existing in text or text in existing for existing in results):
                                    results.append(text)
                            if len(results) >= 3:
                                break
                    
                    # Look for search result descriptions as fallback
                    if len(results) < 3:
                        for span in soup.find_all('span', {'class': ['st', 'aCOpRe']}):
                            text = span.get_text().strip()
                            if text and len(text) > 20:
                                if not any(existing in text or text in existing for existing in results):
                                    results.append(text)
                            if len(results) >= 3:
                                break
                    
                    return results if results else None
                else:
                    # Very basic fallback without BeautifulSoup
                    import re
                    
                    # Look for basic content patterns
                    patterns = [
                        r'<div[^>]*>([^<]{50,400})</div>',
                        r'<span[^>]*>([^<]{50,400})</span>',
                        r'<p[^>]*>([^<]{50,400})</p>'
                    ]
                    
                    results = []
                    for pattern in patterns:
                        matches = re.findall(pattern, response.text)
                        for match in matches[:5]:
                            clean_text = re.sub(r'<[^>]+>', '', match).strip()
                            if clean_text and len(clean_text) > 20:
                                results.append(clean_text)
                        if results:
                            break
                    
                    return results[:3] if results else None
                    
        except Exception as e:
            print(f"{Fore.RED}Google search error: {str(e)}{Style.RESET_ALL}")
            return None
    
    def search_bing_api(self, query):
        """Search using Bing (simple scraping approach)"""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.bing.com/search?q={encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200 and HAS_BS4:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                
                # Look for Bing answer boxes and featured snippets (most reliable)
                answer_selectors = [
                    '.b_ans',
                    '.b_focusTextLarge', 
                    '.b_entityTitle',
                    '.b_factrow',
                    '.ans_nws .na_cnt',
                    '.b_algoup'
                ]
                
                for selector in answer_selectors:
                    elements = soup.select(selector)
                    for elem in elements:
                        text = elem.get_text().strip()
                        if text and len(text) > 20 and len(text) < 600:
                            results.append(text)
                        if len(results) >= 2:
                            break
                    if len(results) >= 2:
                        break
                
                # Look for search snippets if no answer boxes found
                if len(results) < 3:
                    for div in soup.find_all('div', {'class': 'b_caption'}):
                        text = div.get_text().strip()
                        if text and len(text) > 20 and len(text) < 400:
                            if not any(existing in text or text in existing for existing in results):
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
        
        # Try multiple search engines in preferred order
        search_methods = [
            ("Google API", self.search_google_api),      # Use API first if available
            ("Google", self.search_google_simple),       # Fallback to scraping
            ("Bing", self.search_bing_api),
            ("Wikipedia", self.search_wikipedia_api),
            ("DuckDuckGo", self.search_duckduckgo)
        ]
        
        for engine_name, search_func in search_methods:
            try:
                results = search_func(query)
                if results and len(results) > 0:
                    print(f"{Fore.GREEN}*burp* Found knowledge from {engine_name}!{Style.RESET_ALL}")
                    # Streamlined debug output
                    if len(results) > 2:
                        print(f"{Fore.CYAN}Found {len(results)} results{Style.RESET_ALL}")
                    return results
                else:
                    print(f"{Fore.YELLOW}*burp* {engine_name} didn't help...{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}*burp* {engine_name} failed: {str(e)[:50]}...{Style.RESET_ALL}")
                continue
            
            # Reduced delay between search engines for speed
            time.sleep(0.5)  # Reduced from 1 second
        
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
        
        # Updated priority order - avoiding reasoning models that can hang
        preferred_models = [
            "qwen2.5:3b-instruct", # Qwen2.5:3b-instruct (ideal choice)
            "gemma2:2b",            # Gemma2 2B (you have this - reliable!)
            "qwen2:1.5b",           # Qwen 1.5B (you have this - reliable!)
            "phi3:mini",            # Microsoft Phi-3 (reliable fallback)
            "llama3.2:1b",          # Llama 3.2 1B 
            "gemma:2b",             # Google Gemma 2B
            "phi3:3.8b",            # Larger Phi3
            "qwen3:4b",             # Qwen3 4B (can have thinking tokens - lower priority)
            "qwen2.5:3b",           # Alternative qwen2.5 variant
            "qwen2.5",              # Base qwen2.5
            "llama3.2:3b-instruct",
            "gemma2:2b-instruct",
            "tinyllama",
            "gemma:7b"
        ]
        
        # Debug: Print available models for troubleshooting
        print(f"{Fore.CYAN}Available models: {models}{Style.RESET_ALL}")
        
        for preferred in preferred_models:
            # Check for exact match first
            if preferred in models:
                self.current_model = preferred
                print(f"{Fore.GREEN}Selected exact match: {preferred}{Style.RESET_ALL}")
                return preferred
            
            # Check for partial match (e.g., "qwen2.5:3b-instruct-q4" matches "qwen2.5:3b-instruct")
            for available in models:
                if available.startswith(preferred):
                    self.current_model = available
                    print(f"{Fore.GREEN}Selected partial match: {available} (looking for {preferred}){Style.RESET_ALL}")
                    return available
                    
        # Fallback to first available model
        if models:
            self.current_model = models[0]
            print(f"{Fore.YELLOW}Using fallback model: {models[0]}{Style.RESET_ALL}")
            return models[0]
            
        return None
    
    def chat(self, message, conversation_history=None, search_results=None, debug_mode=False):
        """Send chat message to Ollama"""
        try:
            # Build messages array
            messages = [
                {"role": "system", "content": RICK_SYSTEM_PROMPT},
            ]
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Prepare the user message, including search results if available
            user_message = message
            if search_results:
                search_context = "\n\nHere's what I found on the internet:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"{i}. {result}\n"
                user_message = f"{message}\n{search_context}"
                
            messages.append({"role": "user", "content": user_message})
            
            # Only show debug info if debug mode is on
            if debug_mode:
                print(f"{Fore.CYAN}Debug - Sending to Ollama:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Model: {self.current_model}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Messages count: {len(messages)}{Style.RESET_ALL}")
                if search_results:
                    print(f"{Fore.CYAN}Search results included: {len(search_results)}{Style.RESET_ALL}")
            
            payload = {
                "model": self.current_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,  # Slightly lower for more focused responses
                    "top_p": 0.9,
                    "num_ctx": 2048,  # Reduced context window for speed
                    "num_predict": 250,  # Reasonable response length (about 250 words)
                    "stop": ["<think>", "</think>", "<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nYou:"]  # Stop tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=45  # Reduced timeout for speed
            )
            
            if response.status_code == 200:
                full_response = ""
                word_count = 0
                in_thinking = False
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if 'message' in data and 'content' in data['message']:
                                content = data['message']['content']
                                
                                # Filter out thinking tags and content
                                if '<think>' in content:
                                    in_thinking = True
                                    content = content.split('<think>')[0]
                                
                                if '</think>' in content:
                                    in_thinking = False
                                    if '</think>' in content:
                                        content = content.split('</think>')[-1]
                                
                                # Only add content if we're not in a thinking block
                                if not in_thinking and content.strip():
                                    full_response += content
                                    print(content, end='', flush=True)
                                    
                                    # Count words for early stopping
                                    word_count += len(content.split())
                            
                            if data.get('done', False):
                                break
                            
                            # Intelligent stopping - look for sentence endings near limit
                            if word_count > 200:  # Near the limit
                                if any(punct in content for punct in ['.', '!', '?']):
                                    # Found a good stopping point
                                    break
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            continue
                
                # Clean up the response
                cleaned_response = self.clean_response(full_response)
                
                if not cleaned_response.strip():
                    return "*burp* Got an empty response from Ollama. The model might be having a bad day!"
                    
                return cleaned_response
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"{Fore.RED}Ollama error: {error_msg}{Style.RESET_ALL}")
                return f"*burp* Ollama error: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "*burp* Timeout waiting for Ollama response. Try a faster model like phi3:mini!"
        except requests.exceptions.ConnectionError:
            return "*burp* Can't connect to Ollama! Is it running? Try: ollama serve"
        except KeyboardInterrupt:
            return "*burp* Response interrupted! Good thing, the AI was probably rambling anyway."
        except Exception as e:
            print(f"{Fore.RED}Chat error: {str(e)}{Style.RESET_ALL}")
            return f"*burp* Something went wrong: {str(e)}"
    
    def clean_response(self, response):
        """Clean up the response by removing thinking tags and extra whitespace"""
        # Remove any remaining thinking tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'<think>.*', '', response)  # Remove incomplete thinking blocks
        response = re.sub(r'.*</think>', '', response)  # Remove ending of thinking blocks
        
        # Clean up extra whitespace
        response = response.strip()
        
        return response

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
    print(f"{Fore.CYAN}Type 'install <model>' to install a new model{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'search <query>' to force a web search{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type 'debug' to toggle debug mode{Style.RESET_ALL}")
    if internet_available:
        print(f"{Fore.GREEN}🔍 Internet search is active - I'll search when needed!{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.YELLOW}⚠️  No internet - I'll be honest when I don't know things{Style.RESET_ALL}\n")
    
    conversation_history = []
    debug_mode = False
    
    while True:
        try:
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"\n{Fore.GREEN}Rick: {random.choice(RICK_EXITS)}{Style.RESET_ALL}")
                break
                
            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"{Fore.CYAN}Debug mode: {'ON' if debug_mode else 'OFF'}{Style.RESET_ALL}")
                continue
                
            if user_input.lower() == 'models':
                if use_ollama:
                    models = ollama.get_models()
                    print(f"{Fore.CYAN}Available Ollama models:{Style.RESET_ALL}")
                    for model in models:
                        marker = " (current)" if model == ollama.current_model else ""
                        print(f"  - {model}{marker}")
                    print(f"\n{Fore.YELLOW}Recommended models to install:{Style.RESET_ALL}")
                    print(f"  ollama pull qwen2.5:3b-instruct  # Best balance of speed/intelligence")
                    print(f"  ollama pull qwen3:4b            # Newer Qwen model") 
                    print(f"  ollama pull gemma2:2b           # Google's latest")
                else:
                    print(f"{Fore.YELLOW}Using traditional DialoGPT model{Style.RESET_ALL}")
                continue
            
            if user_input.lower().startswith('use '):
                model_name = user_input[4:].strip()
                if use_ollama:
                    available_models = ollama.get_models()
                    if model_name in available_models:
                        ollama.current_model = model_name
                        print(f"{Fore.GREEN}*burp* Switched to {model_name}!{Style.RESET_ALL}")
                        if 'qwen3' in model_name or 'reasoning' in model_name.lower():
                            print(f"{Fore.YELLOW}⚠️  Warning: {model_name} may output thinking tokens and be slower{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}*burp* Model {model_name} not found!{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Available models:{Style.RESET_ALL}")
                        for model in available_models[:10]:  # Show first 10
                            print(f"  - {model}")
                else:
                    print(f"{Fore.RED}*burp* Model switching only works with Ollama!{Style.RESET_ALL}")
                continue
            
            if user_input.lower().startswith('install '):
                model_name = user_input[8:].strip()
                if use_ollama:
                    print(f"{Fore.YELLOW}*burp* Installing {model_name}...{Style.RESET_ALL}")
                    try:
                        result = subprocess.run(['ollama', 'pull', model_name], 
                                              capture_output=True, text=True, timeout=300)
                        if result.returncode == 0:
                            print(f"{Fore.GREEN}*burp* {model_name} installed successfully!{Style.RESET_ALL}")
                            print(f"{Fore.CYAN}Type 'use {model_name}' to switch to it.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}*burp* Failed to install {model_name}: {result.stderr}{Style.RESET_ALL}")
                    except subprocess.TimeoutExpired:
                        print(f"{Fore.RED}*burp* Installation timed out. Large models take time!{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.RED}*burp* Installation error: {e}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}*burp* Model installation only works with Ollama!{Style.RESET_ALL}")
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
                
                # Streamlined search feedback
                if search_results:
                    print(f"{Fore.CYAN}*burp* Got {len(search_results)} results!{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}*burp* No current info found...{Style.RESET_ALL}")
                
            # Show minimal thinking indicator
            print(f"{Fore.GREEN}Rick: {Style.RESET_ALL}", end="", flush=True)
            
            # Get response
            if use_ollama:
                response = ollama.chat(user_input, conversation_history, search_results, debug_mode)
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
