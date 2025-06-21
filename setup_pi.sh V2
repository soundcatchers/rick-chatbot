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
    libxcb1-dev

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
    scipy \
# installed as stand alone sudo see below as didn't work    sqlite3 \
    threading \
    dataclasses \
    typing

# install sqlite 3 as a stand alond command
sudo apt install libsqlite3-dev

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

# Verify startup script was created
if [ -f "start_rick.sh" ]; then
    echo "✅ Startup script created successfully"
else
    echo "❌ Failed to create startup script"
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
echo "Or simply run: $PROJECT_DIR/start_rick.sh"
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
