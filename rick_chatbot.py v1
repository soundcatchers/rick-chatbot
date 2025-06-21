#!/usr/bin/env python3
"""
Rick Sanchez Chatbot for Raspberry Pi 5
Offline LLM with short-term and long-term memory
"""

import json
import os
import sqlite3
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import threading
import time
import random
import re

# Core dependencies
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Represents a memory entry with context and importance"""
    content: str
    timestamp: str
    importance: int  # 1-10 scale
    context_type: str  # 'conversation', 'fact', 'preference', etc.

class ShortTermMemory:
    """Manages recent conversation context"""
    
    def __init__(self, max_entries: int = 20):
        self.max_entries = max_entries
        self.memories: List[MemoryEntry] = []
        self.lock = threading.Lock()
    
    def add_memory(self, content: str, importance: int = 5, context_type: str = "conversation"):
        """Add a new memory entry"""
        with self.lock:
            memory = MemoryEntry(
                content=content,
                timestamp=datetime.datetime.now().isoformat(),
                importance=importance,
                context_type=context_type
            )
            self.memories.append(memory)
            
            # Keep only recent memories
            if len(self.memories) > self.max_entries:
                self.memories = self.memories[-self.max_entries:]
    
    def get_context(self, limit: int = 10) -> str:
        """Get recent conversation context"""
        with self.lock:
            recent = self.memories[-limit:] if self.memories else []
            return "\n".join([m.content for m in recent])
    
    def clear(self):
        """Clear short-term memory"""
        with self.lock:
            self.memories.clear()

class LongTermMemory:
    """Manages persistent memory using SQLite"""
    
    def __init__(self, db_path: str = "rick_memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    context_type TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            conn.commit()
    
    def add_memory(self, memory: MemoryEntry):
        """Store memory in long-term storage"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memories (content, timestamp, importance, context_type)
                    VALUES (?, ?, ?, ?)
                """, (memory.content, memory.timestamp, memory.importance, memory.context_type))
                conn.commit()
    
    def retrieve_memories(self, query: str = None, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                if query:
                    cursor = conn.execute("""
                        SELECT content, timestamp, importance, context_type
                        FROM memories
                        WHERE content LIKE ?
                        ORDER BY importance DESC, timestamp DESC
                        LIMIT ?
                    """, (f"%{query}%", limit))
                else:
                    cursor = conn.execute("""
                        SELECT content, timestamp, importance, context_type
                        FROM memories
                        ORDER BY importance DESC, timestamp DESC
                        LIMIT ?
                    """, (limit,))
                
                memories = []
                for row in cursor.fetchall():
                    memories.append(MemoryEntry(
                        content=row[0],
                        timestamp=row[1],
                        importance=row[2],
                        context_type=row[3]
                    ))
                return memories

class RickPersonality:
    """Rick Sanchez personality traits and response patterns"""
    
    RICK_PHRASES = [
        "*burp* Listen,",
        "Wubba lubba dub dub!",
        "Obviously,",
        "*burp* Science!",
        "Get schwifty!",
        "Nobody exists on purpose,",
        "Life is meaningless,",
        "I'm not a hero,",
        "Existence is pain,",
        "*burp* Whatever,"
    ]
    
    RICK_RESPONSES = {
        "hello": "Yeah, hi, *burp* what do you want?",
        "how are you": "*burp* I'm Rick Sanchez, I'm great, thanks for asking.",
        "what's your name": "I'm Rick, *burp* Rick Sanchez. The smartest man in the universe.",
        "science": "*burp* Finally, someone wants to talk about something interesting.",
        "help": "Help? *burp* Help yourself, that's what I do.",
    }
    
    @staticmethod
    def add_rick_flavor(text: str) -> str:
        """Add Rick's characteristic speech patterns"""
        # Add random burps
        if random.random() < 0.4:
            text = re.sub(r'([.!?])\s+', r'\1 *burp* ', text, count=random.randint(1, 2))
        
        # Add occasional Rick phrases at the beginning
        if random.random() < 0.3:
            phrase = random.choice(RickPersonality.RICK_PHRASES)
            text = f"{phrase} {text}"
        
        # Make responses more Rick-like
        text = text.replace("I think", "Obviously")
        text = text.replace("maybe", "probably")
        text = text.replace("Hello", "Yeah, hi")
        
        return text

class RickChatbot:
    """Main chatbot class with Rick Sanchez personality"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        # Memory systems
        self.short_memory = ShortTermMemory()
        self.long_memory = LongTermMemory()
        self.personality = RickPersonality()
        
        # Conversation state
        self.conversation_count = 0
        self.user_name = "Morty"
        
        logger.info("Initializing Rick Sanchez chatbot...")
        self._load_model()
        self._load_personality_context()
    
    def _load_model(self):
        """Load the language model optimized for Raspberry Pi"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                max_length=100,
                return_full_text=False
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to simple responses if model fails
            self.generator = None
    
    def _load_personality_context(self):
        """Load Rick's personality context into memory"""
        rick_context = [
            "User: Who are you? Rick: I'm Rick Sanchez, *burp* the smartest man in the universe.",
            "User: How are you? Rick: I'm Rick, I'm great, *burp* thanks for asking, I guess.",
            "User: Can you help me? Rick: Help? *burp* Help yourself, that's what I do.",
            "Rick burps frequently and drinks a lot",
            "Rick is extremely intelligent but cynical",
            "Rick considers most people idiots",
            "Rick has a portal gun and travels dimensions",
            "Rick's catchphrase is 'Wubba lubba dub dub'",
        ]
        
        for context in rick_context:
            self.short_memory.add_memory(context, importance=9, context_type="personality")
    
    def _get_rick_response(self, user_input: str) -> str:
        """Generate Rick's response using the model or fallback"""
        user_input_lower = user_input.lower()
        
        # Check for direct personality responses first
        for key, response in self.personality.RICK_RESPONSES.items():
            if key in user_input_lower:
                return response
        
        # Use AI model if available
        if self.generator:
            try:
                # Build context-aware prompt
                context = self.short_memory.get_context(limit=4)
                prompt = f"Rick Sanchez: {context}\nUser: {user_input}\nRick:"
                
                # Generate response
                result = self.generator(
                    prompt,
                    max_length=len(prompt.split()) + 30,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = result[0]['generated_text'].strip()
                
                # Clean up the response
                if "Rick:" in response:
                    response = response.split("Rick:")[-1].strip()
                if "User:" in response:
                    response = response.split("User:")[0].strip()
                
                return self.personality.add_rick_flavor(response)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
        
        # Fallback responses if model fails
        fallback_responses = [
            "*burp* I don't have time for this.",
            "Look, I'm busy saving the universe, *burp* what do you want?",
            "That's not how science works, genius.",
            "*burp* Obviously you don't understand quantum mechanics.",
            "Wubba lubba dub dub! *burp*",
            "I've seen this in dimension C-137, *burp* it doesn't end well.",
            "Science, *burp* it's not magic, it's science!",
            "Listen, existence is pain, *burp* deal with it."
        ]
        
        return random.choice(fallback_responses)
    
    def chat(self, user_input: str) -> str:
        """Main chat method"""
        try:
            # Store user input in memory
            self.short_memory.add_memory(f"User: {user_input}", importance=6)
            
            # Generate Rick's response
            response = self._get_rick_response(user_input)
            
            # Store Rick's response in memory
            self.short_memory.add_memory(f"Rick: {response}", importance=7)
            
            # Occasionally consolidate memories
            self.conversation_count += 1
            if self.conversation_count % 10 == 0:
                self._consolidate_memories()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "*burp* Something went wrong with my neural implants. Try again."
    
    def _consolidate_memories(self):
        """Move important short-term memories to long-term storage"""
        try:
            for memory in self.short_memory.memories:
                if memory.importance >= 8:
                    self.long_memory.add_memory(memory)
            logger.info("Consolidated memories to long-term storage")
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
    
    def set_user_name(self, name: str):
        """Set the user's name"""
        self.user_name = name
        self.short_memory.add_memory(f"User's name is {name}", importance=8, context_type="user_info")
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "short_term_count": len(self.short_memory.memories),
            "conversation_count": self.conversation_count,
            "long_term_available": os.path.exists(self.long_memory.db_path)
        }
    
    def reset_conversation(self):
        """Reset the conversation but keep personality"""
        self.short_memory.clear()
        self._load_personality_context()
        self.conversation_count = 0
        logger.info("Conversation reset")

def main():
    """Main function to run the chatbot"""
    print("🧪 Rick Sanchez Chatbot v1.0")
    print("*burp* Initializing... This might take a moment on the Pi...")
    
    try:
        # Initialize chatbot
        rick = RickChatbot()
        
        print("\n" + "="*50)
        print("Rick: *burp* Great, another person who wants to chat.")
        print("Rick: I'm Rick Sanchez, *burp* the smartest man alive.")
        print("Rick: What do you want? And make it interesting.")
        print("Type 'quit', 'exit', or 'bye' to leave.")
        print("Type 'reset' to clear conversation history.")
        print("Type 'stats' to see memory statistics.")
        print("="*50 + "\n")
        
        # Get user name
        user_name = input("What's your name? ").strip()
        if user_name:
            rick.set_user_name(user_name)
            print(f"Rick: *burp* {user_name}? Fine, whatever.\n")
        
        # Main chat loop
        while True:
            try:
                user_input = input(f"{user_name or 'You'}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Rick: *burp* Finally, some peace and quiet. Later.")
                    break
                elif user_input.lower() == 'reset':
                    rick.reset_conversation()
                    print("Rick: *burp* Fine, we're starting over. What now?")
                    continue
                elif user_input.lower() == 'stats':
                    stats = rick.get_memory_stats()
                    print(f"Rick: *burp* Memory stats: {stats}")
                    continue
                
                # Get Rick's response
                response = rick.chat(user_input)
                print(f"Rick: {response}\n")
                
            except KeyboardInterrupt:
                print("\nRick: *burp* Interrupted? Fine, I'm out.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print("Rick: *burp* Something's broken. Try again.")
                
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        print("*burp* Failed to initialize. Check your dependencies.")

if __name__ == "__main__":
    main()
