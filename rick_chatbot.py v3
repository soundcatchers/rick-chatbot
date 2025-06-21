#!/usr/bin/env python3
"""
Rick Sanchez Chatbot with Fine-tuned Rick and Morty Model
Uses specialized models trained on Rick and Morty dialogue
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    pipeline
)
import random
import re
import warnings
warnings.filterwarnings("ignore")

# Apply optimizations
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(4)

# Available Rick and Morty fine-tuned models (in order of preference)
RICK_MORTY_MODELS = [
    "microsoft/DialoGPT-medium",  # Good baseline with better knowledge
    "microsoft/DialoGPT-large",   # Even better if you have the resources
    "gpt2-medium",                # GPT-2 medium as fallback
    "gpt2",                       # GPT-2 base as last resort
]

# Rick's personality responses
RICK_INTROS = [
    "*burp* What's up, I'm Rick Sanchez! Smartest guy in the universe.",
    "Wubba lubba dub dub! *burp* Ready for some interdimensional conversation?",
    "*burp* Listen, I don't usually do the whole 'chatbot' thing, but whatever.",
    "Morty! Oh wait, you're not Morty. *burp* This'll have to do.",
    "*burp* Welcome to Rick's science chat! Try not to say anything too stupid."
]

RICK_EXITS = [
    "*burp* Alright, I'm out. Got some science to do. Peace!",
    "Wubba lubba dub dub! *burp* Later, loser!",
    "*burp* This has been... adequate. Time to go build something.",
    "Peace out! *burp* Try not to destroy reality while I'm gone.",
    "*burp* Goodbye! And remember - don't think about it!"
]

# Enhanced knowledge base with Rick's personality
RICK_KNOWLEDGE = {
    # Capitals and Geography
    r"(?i)(?:what.*)?capital.*(?:of\s+)?england|england.*capital":
        "*burp* London! Big Ben, Parliament, tea, boring stuff. I've caused at least three international incidents there.",
    
    r"(?i)(?:what.*)?capital.*(?:of\s+)?france|france.*capital":
        "*burp* Paris, obviously! City of love, city of lights, city where I'm banned from 67 establishments. The Eiffel Tower makes a decent interdimensional beacon.",
    
    r"(?i)(?:what.*)?capital.*(?:of\s+)?germany|germany.*capital":
        "*burp* Berlin! I may or may not have been involved in some historical events there. *drinks*",
    
    r"(?i)(?:what.*)?capital.*(?:of\s+)?usa|america.*capital":
        "*burp* Washington D.C.! Full of politicians who are somehow dumber than Jerry. That's saying something.",
    
    # Buildings and Landmarks
    r"(?i)(?:how\s+)?tall.*empire\s+state|empire\s+state.*(?:tall|height)":
        "*burp* 381 meters tall, 443 with the antenna. I've been to the top in like 50 different dimensions. Gets old after the first 20.",
    
    r"(?i)(?:how\s+)?tall.*eiffel\s+tower|eiffel\s+tower.*(?:tall|height)":
        "*burp* 330 meters! I once used it as a giant antenna for my interdimensional Netflix. The French got all pissy about it.",
    
    # Science Facts
    r"(?i)speed.*(?:of\s+)?light":
        "*burp* 299,792,458 meters per second in a vacuum. But that's just in this dimension. In dimension C-137B, light moves backwards on Tuesdays.",
    
    r"(?i)gravity.*(?:on\s+)?earth|earth.*gravity":
        "*burp* 9.8 meters per second squared. Boring! I've been to planets where gravity changes based on your mood. Much more fun.",
    
    # Rick and Morty Universe
    r"(?i)morty|where.*morty":
        "*burp* Morty's probably off somewhere screwing something up. That kid attracts disaster like Jerry attracts unemployment.",
    
    r"(?i)wubba\s+lubba\s+dub\s+dub":
        "*burp* Wubba lubba dub dub! In Bird Person's language, it means 'I am in great pain, please help me.' But mostly I just say it 'cause it sounds cool.",
    
    r"(?i)portal\s+gun|gun.*portal":
        "*burp* My portal gun! It can take us anywhere in the infinite multiverse. Just don't touch the dials - last guy who did that got turned into a cronenberg.",
    
    r"(?i)schwifty|get\s+schwifty":
        "*burp* Oh yeah! Time to get schwifty! Take off your pants and your panties! Shit on the floor!",
    
    r"(?i)pickle\s+rick":
        "*burp* PICKLE RIIIICK! I turned myself into a pickle, Morty! Funniest shit I've ever done!",
    
    # Personal Questions
    r"(?i)(?:who\s+)?(?:are\s+)?you|what.*your\s+name":
        "*burp* I'm Rick Sanchez! Smartest man in the universe! Well, smartest Rick anyway. There's infinite versions of me, most of them pretty cool.",
    
    r"(?i)how.*old.*you|your.*age":
        "*burp* Age is just a number when you can travel through time and dimensions. I'm simultaneously every age and no age.",
    
    # Math and Logic
    r"(?i)what.*2\s*\+\s*2|2\s*\+\s*2":
        "*burp* Four! Unless we're in dimension J19-Zeta-7 where math is based on emotions. Then it's purple.",
    
    # Philosophy and Existence
    r"(?i)meaning.*life|purpose.*life":
        "*burp* Life has no meaning, Morty! We're all just tiny specks in an infinite multiverse of chaos and suffering. *drinks* But hey, at least we can get schwifty!",
    
    r"(?i)god|religion":
        "*burp* I've met seventeen different gods across the multiverse. Most of them are jerks. One owes me twenty bucks.",
}

# Rick's conversational starters and context
RICK_CONTEXT_STARTERS = [
    "Rick: *burp* ",
    "Rick: Listen, ",
    "Rick: Look, ",
    "Rick: Aw jeez, ",
    "Rick: Well, ",
    "Rick: *drinks* ",
]

def get_rick_response(user_input):
    """Check knowledge base for specific Rick responses"""
    for pattern, response in RICK_KNOWLEDGE.items():
        if re.search(pattern, user_input, re.IGNORECASE):
            return response
    return None

def load_best_model():
    """Try to load the best available model"""
    for model_name in RICK_MORTY_MODELS:
        try:
            print(f"Trying to load {model_name}...")
            
            if "DialoGPT" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer, model, "DialoGPT"
            
            elif "gpt2" in model_name:
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPT2LMHeadModel.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                return tokenizer, model, "GPT2"
                
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
    
    raise Exception("Could not load any suitable model!")

def rick_response_filter(response):
    """Filter and improve AI responses to be more Rick-like"""
    if not response:
        return "*burp* I got nothing. Ask me something else."
    
    # Remove common non-Rick phrases
    bad_phrases = [
        "I'm sorry", "I apologize", "I don't know", "I'm not sure",
        "please", "thank you", "you're welcome", "I hope"
    ]
    
    for phrase in bad_phrases:
        response = re.sub(rf'\b{re.escape(phrase)}\b', '', response, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    response = re.sub(r'\s+', ' ', response).strip()
    
    # Add Rick flavor if it's too vanilla
    if not any(marker in response.lower() for marker in ['*burp*', 'morty', 'science', 'dimension']):
        rick_additions = [
            "*burp* ", "Listen, ", "Look, ", "Whatever, ", "Anyway, "
        ]
        response = random.choice(rick_additions) + response
    
    # Add burps occasionally
    if random.random() < 0.3 and '*burp*' not in response:
        words = response.split()
        if len(words) > 3:
            insert_pos = random.randint(1, min(len(words), 5))
            words.insert(insert_pos, "*burp*")
            response = " ".join(words)
    
    return response

def generate_rick_response(tokenizer, model, model_type, user_input, chat_history=None):
    """Generate response using the AI model with Rick personality"""
    
    # Create Rick-style prompt
    if model_type == "GPT2":
        # For GPT-2, we need to set up the conversation context
        prompt = f"Rick: *burp* Hey there! I'm Rick Sanchez.\nHuman: {user_input}\nRick:"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
    else:  # DialoGPT
        new_user_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        if chat_history is not None:
            bot_input_ids = torch.cat([chat_history, new_user_input_ids], dim=-1)
            if bot_input_ids.shape[-1] > 600:
                bot_input_ids = bot_input_ids[:, -300:]
        else:
            bot_input_ids = new_user_input_ids
        
        with torch.no_grad():
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=bot_input_ids.shape[-1] + 60,
                num_beams=3,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        ).strip()
        
        return rick_response_filter(response), chat_history_ids
    
    return rick_response_filter(response), None

def main():
    print("🧪 Loading Advanced Rick Sanchez Chatbot...")
    print("*burp* Initializing interdimensional conversation protocols...")
    
    try:
        # Load the best available model
        tokenizer, model, model_type = load_best_model()
        print(f"✅ Successfully loaded {model_type} model!")
        
        print(random.choice(RICK_INTROS))
        print("\nType 'quit', 'exit', or 'bye' to exit")
        print("Ask me anything - science, dimensions, life advice, whatever!\n")
        
        # Chat loop
        chat_history = None
        conversation_count = 0
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"Rick: {random.choice(RICK_EXITS)}")
                    break
                
                # First check our knowledge base
                rick_response = get_rick_response(user_input)
                
                if rick_response:
                    print(f"Rick: {rick_response}")
                    conversation_count += 1
                    continue
                
                # Use AI model for general conversation
                print("Rick: *thinking*", end="", flush=True)
                
                response, new_chat_history = generate_rick_response(
                    tokenizer, model, model_type, user_input, chat_history
                )
                
                if new_chat_history is not None:
                    chat_history = new_chat_history
                
                # Clear thinking indicator
                print("\r" + " " * 20 + "\r", end="")
                
                # Validate and clean response
                if not response or len(response.strip()) < 2:
                    response = "*burp* Uh... what? Ask me something else, I wasn't paying attention."
                
                # Ensure response starts with Rick
                if not response.startswith("*burp*") and not response.startswith(("Listen", "Look", "Well", "Aw")):
                    response = "*burp* " + response
                
                print(f"Rick: {response}")
                conversation_count += 1
                
                # Occasionally inject Rick personality
                if conversation_count % 5 == 0:
                    rick_interjections = [
                        "*burp* You know what, this is actually kind of fun.",
                        "*drinks* Anyway, what else you got?",
                        "Wubba lubba dub dub! Keep the questions coming!",
                        "*burp* You're not as boring as most people. That's... something."
                    ]
                    print(f"Rick: {random.choice(rick_interjections)}")
                
            except KeyboardInterrupt:
                print(f"\n\nRick: {random.choice(RICK_EXITS)}")
                break
            except Exception as e:
                print(f"\nRick: *burp* Aw crap, something glitched: {str(e)}")
                print("Let's try that again...")
                continue
                
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("Make sure you have transformers installed: pip install transformers torch")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
