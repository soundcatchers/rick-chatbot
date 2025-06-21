#!/bin/bash
# Rick Sanchez Chatbot Setup Script for Raspberry Pi 5
# Run this script to set up the virtual environment and dependencies

set -e

echo "🧪 Rick Sanchez Chatbot Setup"
echo "*burp* Setting up your Pi for some science..."

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "Continuing anyway..."
fi

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

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv rick_env

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source rick_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch CPU version for Raspberry Pi
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other dependencies
echo "🤖 Installing transformers and dependencies..."
pip install \
    transformers \
    tokenizers \
    datasets \
    accelerate \
    numpy \
    scipy

# Create requirements.txt
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.0.0
accelerate>=0.20.0
numpy>=1.24.0
scipy>=1.10.0
EOF

# Download and cache the model
echo "🤖 Pre-downloading the language model..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading DialoGPT-small...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
print('Model downloaded and cached successfully!')
"

# Create the main chatbot Python file
echo "🤖 Creating rick_chatbot.py..."
cat > rick_chatbot.py << 'EOF'
#!/usr/bin/env python3
"""
Rick Sanchez Chatbot
A simple chatbot using transformers with Rick's personality
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

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

def add_rick_flavor(response):
    """Add Rick's personality to responses"""
    rick_additions = [
        "*burp*", "Morty!", "Listen,", "Look,", "Wubba lubba dub dub!",
        "I'm a genius!", "Science!", "*drinks*", "Whatever.",
        "Aw jeez,", "That's... that's not good,", "Interdimensional",
        "quantum", "multiverse", "portal gun", "schwifty"
    ]
    
    # Randomly add Rick flavor
    if random.random() < 0.3:
        addition = random.choice(rick_additions)
        if random.random() < 0.5:
            response = f"{addition} {response}"
        else:
            response = f"{response} {addition}"
    
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
        print("\nType 'quit', 'exit', or 'bye' to exit\n")
        
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
                
                # Encode the user input
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
                    if bot_input_ids.shape[-1] > 800:
                        bot_input_ids = bot_input_ids[:, -400:]
                else:
                    bot_input_ids = new_user_input_ids
                    
                # Generate response
                print("Rick: *thinking*", end="", flush=True)
                with torch.no_grad():
                    chat_history_ids = model.generate(
                        bot_input_ids, 
                        max_length=bot_input_ids.shape[-1] + 50,
                        num_beams=3,
                        no_repeat_ngram_size=2,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
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
                
                if not response:
                    response = "*burp* I don't know what to say to that."
                
                response = add_rick_flavor(response)
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
EOF

chmod +x rick_chatbot.py

# Create startup script
echo "🚀 Creating startup script..."
cat > start_rick.sh << 'EOF'
#!/bin/bash
# Rick Sanchez Chatbot Startup Script

cd "$HOME/rick_chatbot"
source rick_env/bin/activate
python3 rick_chatbot.py
EOF

chmod +x start_rick.sh

# Create a symlink in ~/.local/bin for easy access
echo "🔗 Creating symlink for easy access..."
mkdir -p ~/.local/bin
ln -sf "$PROJECT_DIR/start_rick.sh" ~/.local/bin/start_rick

# Add ~/.local/bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "📝 Added ~/.local/bin to PATH in ~/.bashrc"
    echo "⚠️  Run 'source ~/.bashrc' or restart terminal for PATH changes to take effect"
fi

# Verify required files were created
echo "🔍 Verifying installation..."
REQUIRED_FILES=("rick_chatbot.py" "start_rick.sh" "requirements.txt")
ALL_GOOD=true

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file created successfully"
    else
        echo "❌ Failed to create $file"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = true ]; then
    echo "✅ All required files created successfully!"
else
    echo "❌ Some files are missing. Please check the script output above."
fi

# Create systemd service (optional)
echo "🔧 Creating systemd service..."
cat > rick-chatbot.service << EOF
[Unit]
Description=Rick Sanchez Chatbot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/rick_env/bin
ExecStart=$PROJECT_DIR/rick_env/bin/python $PROJECT_DIR/rick_chatbot.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "🎯 Creating desktop shortcut..."
mkdir -p "$HOME/Desktop"
cat > "$HOME/Desktop/Rick Chatbot.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Rick Chatbot
Comment=Talk to Rick Sanchez
Exec=$PROJECT_DIR/start_rick.sh
Icon=applications-science
Terminal=true
Categories=Science;Education;
EOF

chmod +x "$HOME/Desktop/Rick Chatbot.desktop"

# Performance optimization for Pi
echo "⚡ Optimizing for Raspberry Pi performance..."
cat > optimize_pi.py << 'EOF'
import os
import torch

# Set environment variables for better performance on Pi
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set PyTorch to use fewer threads
torch.set_num_threads(4)

print("Pi optimization settings applied!")
EOF

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "*burp* Congratulations, your Pi is now ready for some interdimensional chatting!"
echo ""
echo "To start the chatbot:"
echo "1. cd $PROJECT_DIR"
echo "2. source rick_env/bin/activate"
echo "3. python3 rick_chatbot.py"
echo ""
echo "Or run one of these commands:"
echo "- ./start_rick.sh (from $PROJECT_DIR)"
echo "- $PROJECT_DIR/start_rick.sh (from anywhere)"
echo "- start_rick (from anywhere, after restarting terminal)"
echo ""
echo "📊 System Requirements Check:"
echo "- RAM: $(free -h | awk '/^Mem:/ {print $2}') (Recommended: 4GB+)"
echo "- Storage: $(df -h $HOME | awk 'NR==2 {print $4}') free"
echo "- CPU: $(nproc) cores detected"
echo ""
echo "🚨 First run will be slower as the model loads into memory."
echo "Subsequent runs will be faster!"
echo ""
echo "Wubba lubba dub dub! 🧪"
