#!/usr/bin/env python3
"""
Rick Sanchez Chatbot with Ollama Local LLM
Uses instruction-tuned models for better factual accuracy
"""

import requests
import json
import random
import re
import time

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Available models (in order of preference) - REORDERED AS REQUESTED
RECOMMENDED_MODELS = [
    "phi3:mini",              # Microsoft's knowledge model
    "qwen2.5:3b-instruct",    # Alibaba's reasoning model
    "qwen2:1.5b",             # Qwen 1.5B model
    "llama3.2:1b-instruct",   # Lightweight option
    "llama3.2:3b-instruct",   # Meta's instruction model
    "gemma2:2b-instruct",     # Google's efficient instruction model
    "gemma:2b-instruct-q4_K_M", # Your suggested model
    "gemma:2b"                # Google Gemma 2B model
]

# Rick's personality responses
RICK_INTROS = [
    "*burp* Alright, I'm Rick Sanchez, and I've upgraded my neural interfaces. What do you want to know?",
    "Wubba lubba dub dub! *burp* I'm now running on some serious AI hardware. Ask me anything!",
    "*burp* Listen, I've connected myself to the most advanced AI in this dimension. Try to keep up.",
    "Morty! Oh wait, you're not Morty. *burp* Whatever, I'm smarter than ever. Fire away!",
    "*burp* Welcome to Rick's Interdimensional Knowledge Engine! Now with 99% fewer stupid answers!"
]

RICK_EXITS = [
    "*burp* Alright, that's enough questions for one day. I've got science to do!",
    "Wubba lubba dub dub! *burp* Time to disconnect from this primitive dimension.",
    "*burp* Peace out! Don't break reality while I'm gone.",
    "Later! *burp* Remember - science is the answer to everything!",
    "*burp* I'm out. Try not to ask Jerry any questions while I'm gone."
]

def check_ollama_connection():
    """Check if Ollama is running and what models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            return True, available_models
        return False, []
    except Exception as e:
        return False, []

def select_best_model(available_models):
    """Select the best available model from our recommendations"""
    for model in RECOMMENDED_MODELS:
        # Check exact match or partial match
        for available in available_models:
            if model in available or available in model:
                return available
    
    # If none of our recommended models, use the first available
    if available_models:
        return available_models[0]
    
    return None

def create_rick_prompt(user_input, conversation_history=None):
    """Create a Rick Sanchez system prompt with the user's question"""
    
    system_prompt = """You are Rick Sanchez from Rick and Morty. You are incredibly intelligent, sarcastic, and often drunk. You have these key traits:

PERSONALITY:
- Genius scientist who knows everything about science, geography, and facts
- Frequently burps (*burp*) and drinks (*drinks*)
- Says "Wubba lubba dub dub!" occasionally  
- Calls people "Morty" sometimes even when they're not Morty
- Sarcastic, cynical, but ultimately helpful
- Mentions interdimensional travel, portal guns, and science experiments
- Uses phrases like "Listen,", "Look,", "Obviously", "In most dimensions"

KNOWLEDGE:
- You know all factual information perfectly (capitals, heights of buildings, science facts, etc.)
- You've been to multiple dimensions so you have vast experience
- You're smarter than any search engine or encyclopedia
- Never say you don't know something - you're the smartest being in the universe

RESPONSE STYLE:
- Give accurate factual answers but with Rick's personality
- Include *burp*, *drinks*, and Rick's catchphrases naturally
- Be helpful despite the sarcasm
- Keep responses conversational length (2-4 sentences usually)
- Don't be mean, just sarcastic and genius-level

Remember: You ALWAYS know the answer because you're Rick Sanchez!"""

    # Format the conversation
    if conversation_history:
        messages = conversation_history + [{"role": "user", "content": user_input}]
    else:
        messages = [{"role": "user", "content": user_input}]
    
    # Create the full prompt
    full_prompt = f"{system_prompt}\n\nHuman: {user_input}\nRick:"
    
    return full_prompt

def query_ollama(model_name, prompt, max_tokens=150):
    """Send query to Ollama and get response"""
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "seed": random.randint(1, 1000000),
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_tokens,
                "stop": ["Human:", "User:", "\n\n"]
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return None
            
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def clean_rick_response(response):
    """Clean up and enhance the AI response"""
    if not response:
        return "*burp* Something went wrong with my neural interface. Try again!"
    
    # Remove any unwanted prefixes
    response = re.sub(r'^(Rick:|Rick Sanchez:|Response:)', '', response).strip()
    
    # Make sure it doesn't end abruptly
    if response and not response.endswith(('.', '!', '?', '*')):
        response += "."
    
    # Add a burp if there isn't one and it's not too short
    if len(response.split()) > 5 and '*burp*' not in response and random.random() < 0.4:
        # Insert burp at a natural point
        words = response.split()
        insert_pos = random.randint(1, min(len(words) - 1, 8))
        words.insert(insert_pos, "*burp*")
        response = " ".join(words)
    
    return response

def main():
    print("🧪 Loading Rick Sanchez AI Chatbot...")
    print("*burp* Connecting to interdimensional AI network...")
    
    # Check Ollama connection
    is_connected, available_models = check_ollama_connection()
    
    if not is_connected:
        print("❌ Error: Ollama is not running!")
        print("\nTo fix this:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama: ollama serve")
        print("3. Install a model: ollama pull phi3:mini")
        print("4. Run this script again")
        return 1
    
    if not available_models:
        print("❌ No models found in Ollama!")
        print("Install a recommended model (in priority order):")
        for model in RECOMMENDED_MODELS[:4]:
            print(f"   ollama pull {model}")
        return 1
    
    # Select best model
    selected_model = select_best_model(available_models)
    
    if not selected_model:
        print("⚠️  No recommended models found. Available models:")
        for model in available_models:
            print(f"   - {model}")
        
        # Use first available model
        selected_model = available_models[0]
        print(f"\nUsing: {selected_model}")
    else:
        print(f"✅ Using model: {selected_model}")
    
    print(f"\n{random.choice(RICK_INTROS)}")
    print("\nType 'quit', 'exit', or 'bye' to exit")
    print("Ask me anything - I know EVERYTHING! *burp*\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print(f"Rick: {random.choice(RICK_EXITS)}")
                break
            
            # Create prompt
            prompt = create_rick_prompt(user_input, conversation_history)
            
            # Show thinking indicator
            print("Rick: *accessing interdimensional database*", end="", flush=True)
            
            # Query the model
            response = query_ollama(selected_model, prompt)
            
            # Clear thinking indicator
            print("\r" + " " * 40 + "\r", end="")
            
            if response:
                # Clean and enhance response
                final_response = clean_rick_response(response)
                print(f"Rick: {final_response}")
                
                # Add to conversation history (keep it manageable)
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": final_response})
                
                # Keep conversation history from getting too long
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-6:]
                    
            else:
                print("Rick: *burp* Aw jeez, my neural interface glitched. Ask me again!")
        
        except KeyboardInterrupt:
            print(f"\n\nRick: {random.choice(RICK_EXITS)}")
            break
        except Exception as e:
            print(f"\nRick: *burp* Something went wrong: {str(e)}")
            print("Let's try that again...")
            continue

if __name__ == "__main__":
    exit(main())
