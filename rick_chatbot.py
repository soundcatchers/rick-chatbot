#!/usr/bin/env python3
"""
Rick Sanchez Chatbot - Enhanced Edition
Supports both Ollama and traditional transformers
"""

import os
import sys
import json
import requests
import random
import time
from colorama import init, Fore, Style

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

Respond as Rick would, including his speech patterns and personality, but keep responses helpful and engaging."""

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
            "phi3:mini",         # Microsoft Phi-3 (your #1 choice)
            "qwen2:1.5b",        # Qwen 1.5B (your #2 choice)  
            "llama3.2:1b",       # Llama 3.2 1B (your #3 choice)
            "gemma:2b",          # Google Gemma 2B (your #4 choice)
            "qwen2.5:3b-instruct",
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
    
    def chat(self, message, conversation_history=None):
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
                
            payload["messages"].append({"role": "user", "content": message})
            
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

def traditional_chat(tokenizer, model, message, conversation_history=None):
    """Chat using traditional transformers model"""
    try:
        import torch
        
        # Create Rick-style prompt
        rick_prompt = f"Human: {message}\nRick: *burp*"
        
        # Encode input
        inputs = tokenizer.encode(rick_prompt, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
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
    print(f"{Fore.CYAN}🧪 Rick Sanchez Chatbot - Enhanced Edition{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}*burp* Loading advanced AI systems...{Style.RESET_ALL}")
    
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
    print(f"{Fore.CYAN}Type 'models' to see available models{Style.RESET_ALL}\n")
    
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
                
            # Show thinking indicator
            print(f"{Fore.YELLOW}Rick: *burp* Let me think...{Style.RESET_ALL}", end="", flush=True)
            
            # Get response
            if use_ollama:
                print(f"\r{Fore.GREEN}Rick: {Style.RESET_ALL}", end="", flush=True)
                response = ollama.chat(user_input, conversation_history)
                print()  # New line after streaming response
            else:
                response = traditional_chat(tokenizer, traditional_model, user_input, conversation_history)
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
