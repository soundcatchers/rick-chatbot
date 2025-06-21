#!/usr/bin/env python3
"""
Improved Rick Sanchez Chatbot
Better responses with factual knowledge and Rick's personality
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import re

# Apply Pi optimizations
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(4)

# Rick's personality responses
RICK_INTROS = [
    "*burp* Oh great, another human wants to chat. What do you want?",
    "Wubba lubba dub dub! *burp* What's up?",
    "*burp* Listen, I'm a genius scientist, not a chatbot, but... whatever.",
    "Morty! Oh wait, you're not Morty. *burp* What do you need?",
    "*burp* Welcome to Rick's interdimensional chat experience!"
]

RICK_EXITS = [
    "*burp* Finally! I've got science to do. Peace out!",
    "Wubba lubba dub dub! *burp* See ya later!",
    "*burp* This conversation is over. I've got portals to build!",
    "Later! *burp* Try not to destroy the universe while I'm gone.",
    "*burp* Peace among worlds! And by that, I mean... well, you know."
]

# Basic knowledge base for common questions
KNOWLEDGE_BASE = {
    # Buildings and landmarks
    r"(?i)empire state building.*tall|height.*empire state": 
        "*burp* The Empire State Building? 381 meters tall, 443 with the antenna. I've been there in like 50 different dimensions. Boring.",
    
    r"(?i)eiffel tower.*where|where.*eiffel tower":
        "*burp* Paris, France, Morty! It's 330 meters tall. I once used it as a giant antenna for my interdimensional radio. Good times.",
    
    r"(?i)capital.*england|england.*capital":
        "*burp* London, obviously. I've blown up Parliament there in at least 3 dimensions. *drinks*",
    
    r"(?i)capital.*france|france.*capital":
        "*burp* Paris. City of lights, city of romance, city where I got banned from 47 cafés. Whatever.",
    
    r"(?i)capital.*germany|germany.*capital":
        "*burp* Berlin. Used to be divided, now it's not. I may have had something to do with the wall coming down. Don't ask.",
    
    # Science questions
    r"(?i)speed.*light":
        "*burp* 299,792,458 meters per second in a vacuum. But with my portal gun, I can go faster. Physics is more like... guidelines.",
    
    r"(?i)gravity.*earth":
        "*burp* 9.8 meters per second squared. But I've been to planets where gravity goes sideways. Much more fun.",
    
    # Personal questions
    r"(?i)how.*you|what.*you":
        "*burp* I'm Rick Sanchez! Smartest man in the universe! Well, in most universes. There's this one Rick who's slightly smarter but he's a jerk.",
    
    r"(?i)favorite.*color":
        "*burp* Lab coat white and portal green. The colors of SCIENCE, Morty!",
    
    # Default responses for unmatched factual questions
    r"(?i)what.*|where.*|when.*|who.*|how.*|why.*":
        "*burp* Look, I'm not Wikipedia. I'm a genius scientist, not a search engine. But I probably know the answer anyway."
}

def get_rick_response(user_input):
    """Check if we have a specific Rick response for this input"""
    for pattern, response in KNOWLEDGE_BASE.items():
        if re.search(pattern, user_input):
            return response
    return None

def add_rick_flavor(response, intensity=0.3):
    """Add Rick's personality to responses with better context"""
    # Don't modify if response already has Rick elements
    if any(word in response.lower() for word in ['*burp*', 'morty', 'wubba', 'science']):
        return response
    
    rick_starts = ["*burp*", "Listen,", "Look,", "Aw jeez,", "Well,"]
    rick_ends = ["*burp*", "Morty!", "Whatever.", "*drinks*", "Science!"]
    rick_modifiers = ["obviously", "clearly", "scientifically speaking", "in most dimensions"]
    
    # Add Rick flavor based on intensity
    if random.random() < intensity:
        if random.random() < 0.4:
            response = f"{random.choice(rick_starts)} {response}"
        if random.random() < 0.3:
            response = f"{response} {random.choice(rick_ends)}"
        if random.random() < 0.2:
            words = response.split()
            if len(words) > 3:
                insert_pos = random.randint(1, len(words)-1)
                words.insert(insert_pos, random.choice(rick_modifiers))
                response = " ".join(words)
    
    return response

def main():
    print("🧪 Loading Rick Sanchez Chatbot...")
    print("*burp* Getting ready for some science...")
    
    try:
        # Load the model and tokenizer
        print("Loading DialoGPT model...")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded! Ready to chat!")
        print(random.choice(RICK_INTROS))
        print("\nType 'quit', 'exit', or 'bye' to exit")
        print("Ask me about science, places, or just chat!\n")
        
        # Chat loop
        chat_history_ids = None
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"Rick: {random.choice(RICK_EXITS)}")
                    break
                
                # Check for specific Rick knowledge first
                rick_response = get_rick_response(user_input)
                
                if rick_response:
                    print(f"Rick: {rick_response}")
                    continue
                
                # Use the AI model for general conversation
                new_user_input_ids = tokenizer.encode(
                    user_input + tokenizer.eos_token, 
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                )
                
                # Append to chat history
                if chat_history_ids is not None:
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                    # Keep conversation history manageable
                    if bot_input_ids.shape[-1] > 600:
                        bot_input_ids = bot_input_ids[:, -300:]
                else:
                    bot_input_ids = new_user_input_ids
                    
                # Generate response
                print("Rick: *thinking*", end="", flush=True)
                with torch.no_grad():
                    chat_history_ids = model.generate(
                        bot_input_ids, 
                        max_length=bot_input_ids.shape[-1] + 40,
                        num_beams=2,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                        attention_mask=torch.ones_like(bot_input_ids)
                    )
                
                # Clear the thinking indicator
                print("\r" + " " * 20 + "\r", end="")
                
                # Decode and print response
                response = tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                    skip_special_tokens=True
                ).strip()
                
                if not response or len(response) < 3:
                    response = "I'm not sure what to say to that."
                
                # Clean up the response
                response = response.replace("  ", " ").strip()
                
                # Add Rick personality more thoughtfully
                response = add_rick_flavor(response, intensity=0.4)
                
                print(f"Rick: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\nRick: {random.choice(RICK_EXITS)}")
                break
            except Exception as e:
                print(f"\nRick: *burp* Aw jeez, something went wrong: {str(e)}")
                print("Let's try again...")
                continue
                
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Make sure you've run the setup script and have internet connection.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
