#!/bin/bash
# Rick Sanchez Chatbot Setup Script for Raspberry Pi 5
# Now with Ollama support for better AI models!
# Run this script to set up both traditional transformers and Ollama options

set -e

echo "🧪 Rick Sanchez Chatbot Setup (Enhanced Edition)"
echo "*burp* Setting up your Pi for some REAL science with multiple AI backends..."

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "Continuing anyway..."
fi

# Function to check available RAM
check_ram() {
    RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
    echo "📊 Detected RAM: ${RAM_GB}GB"
    if [ "$RAM_GB" -lt 4 ]; then
        echo "⚠️  Warning: Less than 4GB RAM detected. Ollama models may run slowly."
        echo "Consider using lighter models or the traditional transformers approach."
    fi
}

check_ram

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libsqlite3-dev

# Create project directory
PROJECT_DIR="$HOME/rick_chatbot"
echo "📁 Creating project directory at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Install Ollama
echo "🤖 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Downloading and installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✅ Ollama installed successfully!"
else
    echo "✅ Ollama is already installed"
fi

# Start Ollama service in background
echo "🚀 Starting Ollama service..."
sudo systemctl enable ollama 2>/dev/null || echo "Note: systemd service not available, will start manually"
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Starting Ollama server in background..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 5
    echo "✅ Ollama server started"
else
    echo "✅ Ollama server already running"
fi

# Wait a bit more for Ollama to fully initialize
echo "⏳ Waiting for Ollama to fully initialize..."
sleep 3

# Function to check if Ollama is ready
check_ollama_ready() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama API is ready"
            return 0
        fi
        echo "⏳ Waiting for Ollama API... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    echo "❌ Ollama API not responding after $max_attempts attempts"
    return 1
}

# Check if Ollama is ready
if ! check_ollama_ready; then
    echo "⚠️  Ollama may not be fully ready, but continuing with model installation..."
fi

# Install recommended Ollama models
echo "🧠 Installing recommended AI models..."
echo "This may take several minutes depending on your internet connection..."

# Function to pull model with error handling and retries
pull_model() {
    local model_name=$1
    local description=$2
    local max_retries=3
    local retry=1
    
    echo "📥 Pulling $model_name ($description)..."
    
    while [ $retry -le $max_retries ]; do
        if [ $retry -gt 1 ]; then
            echo "🔄 Retry $retry/$max_retries for $model_name..."
            sleep 5
        fi
        
        if timeout 600 ollama pull "$model_name" 2>&1; then
            echo "✅ $model_name installed successfully"
            return 0
        else
            echo "❌ Attempt $retry failed for $model_name"
            ((retry++))
        fi
    done
    
    echo "❌ Failed to install $model_name after $max_retries attempts"
    return 1
}

# Install models (REORDERED AS REQUESTED)
MODELS_INSTALLED=0

echo "🔍 Installing models in preferred order..."

# Try Phi3 mini first (Microsoft model - your #1 choice)
if pull_model "phi3:mini" "Microsoft Phi-3 mini model"; then
    MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
    DEFAULT_MODEL="phi3:mini"
fi

# Try Qwen 1.5B second (your #2 choice)
if pull_model "qwen2:1.5b" "Qwen 1.5B model"; then
    MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
    [ -z "$DEFAULT_MODEL" ] && DEFAULT_MODEL="qwen2:1.5b"
fi

# Try Llama 3.2 1B third (your #3 choice)
if pull_model "llama3.2:1b" "Meta Llama 3.2 1B model"; then
    MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
    [ -z "$DEFAULT_MODEL" ] && DEFAULT_MODEL="llama3.2:1b"
fi

# Try Gemma 2B last (your #4 choice)
if pull_model "gemma:2b" "Google Gemma 2B model"; then
    MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
    [ -z "$DEFAULT_MODEL" ] && DEFAULT_MODEL="gemma:2b"
fi

# If none of the preferred models worked, try some additional fallback options
if [ $MODELS_INSTALLED -eq 0 ]; then
    echo "⚠️  Primary models failed, trying fallback options..."
    
    # Try basic models that should definitely exist
    fallback_models=(
        "tinyllama"
        "phi3:3.8b"
        "gemma:7b"
        "llama3:8b"
    )
    
    for model in "${fallback_models[@]}"; do
        if pull_model "$model" "Fallback model"; then
            MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
            [ -z "$DEFAULT_MODEL" ] && DEFAULT_MODEL="$model"
            break
        fi
    done
fi

# If still no models, let's check what's available
if [ $MODELS_INSTALLED -eq 0 ]; then
    echo "🔍 No models installed successfully. Let's check what's available..."
    echo "Available models from Ollama library:"
    
    # Try to get a list of available models (this might not work on all systems)
    curl -s https://ollama.ai/library 2>/dev/null | grep -o 'href="/library/[^"]*"' | cut -d'"' -f2 | head -10 || true
    
    echo ""
    echo "⚠️  You can manually install models later using:"
    echo "   ollama pull <model_name>"
    echo ""
    echo "Recommended models to try manually (in priority order):"
    echo "   ollama pull phi3:mini      # Microsoft Phi-3 (recommended #1)"
    echo "   ollama pull qwen2:1.5b     # Qwen 1.5B (recommended #2)"
    echo "   ollama pull llama3.2:1b    # Llama 3.2 1B (recommended #3)"
    echo "   ollama pull gemma:2b       # Google Gemma 2B (recommended #4)"
fi

echo "✅ Installed $MODELS_INSTALLED Ollama models"
if [ -n "$DEFAULT_MODEL" ]; then
    echo "🎯 Default model set to: $DEFAULT_MODEL"
fi

# Create Python virtual environment for traditional approach
echo "🐍 Creating Python virtual environment..."
python3 -m venv rick_env

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source rick_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies for both approaches
echo "🔧 Installing Python dependencies..."
pip install \
    requests \
    numpy \
    colorama

# For traditional transformers approach (optional)
echo "🤖 Installing transformers (for traditional approach)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install \
    transformers \
    tokenizers \
    datasets \
    accelerate \
    scipy

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
requests>=2.28.0
colorama>=0.4.6
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.0.0
accelerate>=0.20.0
scipy>=1.10.0
EOF

# Download and cache the traditional model (optional)
echo "🤖 Pre-downloading traditional model (backup option)..."
python3 -c "
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('Downloading DialoGPT-small as backup...')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
    print('Traditional model downloaded successfully!')
except Exception as e:
    print(f'Note: Traditional model download failed: {e}')
    print('Ollama models will be used instead.')
"

# Create the enhanced chatbot Python file with updated model handling
echo "🤖 Creating enhanced rick_chatbot.py..."
cat > rick_chatbot.py << 'EOF'
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
            print(f"{Fore.RED}❌ No AI
