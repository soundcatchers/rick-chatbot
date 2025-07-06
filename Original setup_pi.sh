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
    libsqlite3-dev \
    beautifulsoup4 \
    python-dotenv
    
# Create project directory
PROJECT_DIR="$HOME/rick_chatbot"
echo "📁 Creating project directory at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create Python virtual environment EARLY (before other operations)
echo "🐍 Creating Python virtual environment..."
if [ -d "rick_env" ]; then
    echo "⚠️  Virtual environment already exists, removing old one..."
    rm -rf rick_env
fi

# Create virtual environment with explicit error checking
if ! python3 -m venv rick_env; then
    echo "❌ Failed to create virtual environment"
    echo "Checking python3-venv installation..."
    if ! python3 -c "import venv" 2>/dev/null; then
        echo "❌ python3-venv module not properly installed"
        echo "Trying to install it again..."
        sudo apt install -y python3-venv python3-dev
        if ! python3 -m venv rick_env; then
            echo "❌ Still failed to create virtual environment"
            exit 1
        fi
    else
        echo "❌ Unknown error creating virtual environment"
        exit 1
    fi
fi

# Verify virtual environment was created
if [ ! -d "rick_env" ] || [ ! -f "rick_env/bin/activate" ]; then
    echo "❌ Virtual environment creation failed - directory or activate script missing"
    exit 1
fi

echo "✅ Virtual environment created successfully"

# Test activation before proceeding
echo "⚡ Testing virtual environment activation..."
if ! source rick_env/bin/activate; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment activation failed - VIRTUAL_ENV not set"
    exit 1
fi

echo "✅ Virtual environment activated successfully"
echo "📍 Virtual environment path: $VIRTUAL_ENV"

# Upgrade pip in virtual environment
echo "📦 Upgrading pip in virtual environment..."
if ! pip install --upgrade pip; then
    echo "❌ Failed to upgrade pip"
    exit 1
fi

# Install basic Python dependencies first
echo "🔧 Installing basic Python dependencies..."
if ! pip install requests colorama numpy; then
    echo "❌ Failed to install basic dependencies"
    exit 1
fi

echo "✅ Basic Python dependencies installed"

# Download the main chatbot file directly from GitHub
echo "📥 Downloading rick_chatbot.py from GitHub..."
if curl -L -o rick_chatbot.py "https://raw.githubusercontent.com/soundcatchers/rick-chatbot/main/rick_chatbot.py"; then
    echo "✅ rick_chatbot.py downloaded successfully!"
    chmod +x rick_chatbot.py
else
    echo "❌ Failed to download rick_chatbot.py from GitHub"
    echo "Please check your internet connection and try again."
    exit 1
fi

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

# Install PyTorch and transformers (optional, for traditional approach)
echo "🤖 Installing PyTorch and transformers (for traditional approach)..."
echo "This may take a while..."

# Install PyTorch first
if ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
    echo "⚠️  Failed to install PyTorch, trying alternative approach..."
    if ! pip install torch torchvision torchaudio; then
        echo "⚠️  PyTorch installation failed, continuing without it..."
        echo "Traditional transformer models may not work."
    fi
fi

# Install transformers and related packages
if ! pip install transformers tokenizers datasets accelerate scipy; then
    echo "⚠️  Failed to install some transformer packages"
    echo "Continuing with basic installation..."
fi

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

# Verify the chatbot file exists
if [ ! -f "rick_chatbot.py" ]; then
    echo "❌ Error: rick_chatbot.py not found after download!"
    echo "Please check your internet connection and GitHub repository."
    exit 1
fi

echo "✅ rick_chatbot.py verified in project directory"

# Create start script with better error handling
echo "🚀 Creating start script..."
cat > start_rick.sh << 'EOF'
#!/bin/bash
cd ~/rick_chatbot

# Check if rick_chatbot.py exists
if [ ! -f "rick_chatbot.py" ]; then
    echo "❌ Error: rick_chatbot.py not found!"
    echo "Please run the setup script again or manually copy the file."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "rick_env" ] || [ ! -f "rick_env/bin/activate" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please run the setup script again."
    exit 1
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source rick_env/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Failed to activate virtual environment!"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Check if Ollama is running, start if needed
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🚀 Starting Ollama server..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
    echo "✅ Ollama server started"
fi

echo "🧪 Starting Rick Sanchez Chatbot..."
python3 rick_chatbot.py
EOF

chmod +x start_rick.sh

# Create systemd service file (optional)
echo "🔧 Creating systemd service file..."
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

# Deactivate virtual environment for final summary
deactivate 2>/dev/null || true

# Final setup summary
echo ""
echo "🎉 Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Project directory: $PROJECT_DIR"
echo "🐍 Virtual environment: $PROJECT_DIR/rick_env"
echo "🧠 Installed models: $MODELS_INSTALLED"
if [ -n "$DEFAULT_MODEL" ]; then
    echo "🎯 Default model: $DEFAULT_MODEL"
fi
echo ""
echo "🚀 To start the chatbot:"
echo "   cd $PROJECT_DIR"
echo "   ./start_rick.sh"
echo ""
echo "Or manually:"
echo "   cd $PROJECT_DIR"
echo "   source rick_env/bin/activate"
echo "   python3 rick_chatbot.py"
echo ""

if [ $MODELS_INSTALLED -eq 0 ]; then
    echo "⚠️  No Ollama models were installed successfully."
    echo "You can install them manually with:"
    echo "   ollama pull phi3:mini"
    echo "   ollama pull qwen2:1.5b"
    echo "   ollama pull llama3.2:1b"
    echo "   ollama pull gemma:2b"
    echo ""
fi

echo "🔧 To install as a system service:"
echo "   sudo cp rick-chatbot.service /etc/systemd/system/"
echo "   sudo systemctl enable rick-chatbot"
echo "   sudo systemctl start rick-chatbot"
echo ""
echo "🔍 To verify virtual environment:"
echo "   cd $PROJECT_DIR"
echo "   source rick_env/bin/activate"
echo "   which python3"
echo "   pip list"
echo ""
echo "*burp* Wubba lubba dub dub! Your Rick chatbot is ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
