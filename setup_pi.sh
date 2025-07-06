#!/bin/bash
# Rick Sanchez Chatbot Setup Script for Raspberry Pi 5 - Enhanced Edition
# Now with Google Search API support and qwen2.5:3b-instruct as default!
# Run this script to set up both traditional transformers and Ollama options

set -e

echo "🧪 Rick Sanchez Chatbot Setup - Enhanced Edition with Internet Search"
echo "*burp* Setting up your Pi for some REAL science with multiple AI backends and web search..."

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
        echo "⚠️  Warning: Less than 4GB RAM detected. Larger models may run slowly."
        echo "Consider using lighter models or traditional transformers approach."
    elif [ "$RAM_GB" -ge 8 ]; then
        echo "✅ 8GB+ RAM detected - perfect for qwen2.5:3b-instruct!"
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
if ! pip install requests colorama numpy python-dotenv; then
    echo "❌ Failed to install basic dependencies"
    exit 1
fi

echo "✅ Basic Python dependencies installed"

# Install BeautifulSoup for web scraping
echo "🔧 Installing web scraping dependencies..."
if ! pip install beautifulsoup4 lxml; then
    echo "⚠️  Failed to install BeautifulSoup, web scraping may be limited"
fi

# Download the main chatbot file directly from GitHub
echo "📥 Downloading enhanced rick_chatbot.py from GitHub..."
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

# Install recommended Ollama models in NEW PRIORITY ORDER
echo "🧠 Installing recommended AI models in priority order..."
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
        
        if timeout 900 ollama pull "$model_name" 2>&1; then
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

# Install models in NEW PRIORITY ORDER
MODELS_INSTALLED=0

echo "🔍 Installing models in NEW priority order (qwen2.5:3b-instruct first)..."

# Try qwen2.5:3b-instruct FIRST (NEW #1 choice)
if pull_model "qwen2.5:3b-instruct" "Qwen2.5 3B Instruct - Recommended model"; then
    MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
    DEFAULT_MODEL="qwen2.5:3b-instruct"
    echo "🎯 qwen2.5:3b-instruct installed as primary model!"
fi

# Try alternative qwen variants
if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "qwen2.5:3b" "Qwen2.5 3B base model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="qwen2.5:3b"
    fi
fi

if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "qwen2.5" "Qwen2.5 base model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="qwen2.5"
    fi
fi

# Try Phi3 mini second (fallback #1)
if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "phi3:mini" "Microsoft Phi-3 mini model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="phi3:mini"
    fi
fi

# Try Qwen 1.5B third (fallback #2)
if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "qwen2:1.5b" "Qwen 1.5B model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="qwen2:1.5b"
    fi
fi

# Try Llama 3.2 1B fourth (fallback #3)
if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "llama3.2:1b" "Meta Llama 3.2 1B model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="llama3.2:1b"
    fi
fi

# Try Gemma 2B last (fallback #4)
if [ $MODELS_INSTALLED -eq 0 ]; then
    if pull_model "gemma:2b" "Google Gemma 2B model"; then
        MODELS_INSTALLED=$((MODELS_INSTALLED + 1))
        DEFAULT_MODEL="gemma:2b"
    fi
fi

# If none of the preferred models worked, try additional fallback options
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
    echo "Recommended models to try manually (in NEW priority order):"
    echo "   ollama pull qwen2.5:3b-instruct  # NEW #1 choice - best balance"
    echo "   ollama pull phi3:mini            # Fast alternative"
    echo "   ollama pull qwen2:1.5b           # Lightweight option"
    echo "   ollama pull llama3.2:1b          # Meta's model"
    echo "   ollama pull gemma:2b             # Google's model"
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

# Create requirements.txt with NEW dependencies
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
requests>=2.28.0
colorama>=0.4.6
numpy>=1.24.0
python-dotenv>=1.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
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

# Create .env example file
echo "🔑 Creating .env example file for Google Search API..."
cat > .env.example << EOF
# Google Custom Search API Configuration
# Get these from: https://developers.google.com/custom-search/v1/introduction

# Your Google API Key (from Google Cloud Console)
GOOGLE_API_KEY=your_google_api_key_here

# Your Custom Search Engine ID 
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# Optional: Additional configuration
# OLLAMA_HOST=http://localhost:11434
# MAX_SEARCH_RESULTS=5
# DEBUG_MODE=false
EOF

# Check if .env already exists
if [ ! -f ".env" ]; then
    echo "📋 .env file not found - creating template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your Google API credentials for web search functionality"
else
    echo "✅ .env file already exists"
fi

# Create start script with better error handling and Google API info
echo "🚀 Creating enhanced start script..."
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

# Check for Google API configuration
if [ -f ".env" ]; then
    if grep -q "your_google_api_key_here" .env 2>/dev/null; then
        echo "⚠️  Google Search API not configured"
        echo "📋 Edit .env file to add your Google API credentials for better search results"
        echo "🔗 See: https://developers.google.com/custom-search/v1/introduction"
    else
        echo "🔑 Google Search API configuration detected"
    fi
else
    echo "⚠️  No .env file found - web search will use fallback methods"
fi

echo "🧪 Starting Rick Sanchez Chatbot - Enhanced Edition..."
python3 rick_chatbot.py
EOF

chmod +x start_rick.sh

# Create systemd service file (optional)
echo "🔧 Creating systemd service file..."
cat > rick-chatbot.service << EOF
[Unit]
Description=Rick Sanchez Chatbot - Enhanced Edition
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

# Create Google API setup instructions
echo "📖 Creating Google API setup guide..."
cat > GOOGLE_API_SETUP.md << 'EOF'
# Google Search API Setup Guide

## Step 1: Install Required Package (Already Done)
```bash
pip install python-dotenv
```

## Step 2: Get Google API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Go to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **API Key**
5. Copy the API key

## Step 3: Enable Custom Search API

1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Custom Search API"
3. Click on it and press **Enable**

## Step 4: Create Custom Search Engine

1. Go to [Google Custom Search Engine](https://cse.google.com/cse/)
2. Click **Add** to create a new search engine
3. For "Sites to search", enter `*` (to search the entire web)
4. Give it a name like "Rick Chatbot Search"
5. Click **Create**
6. Copy the **Search Engine ID** (looks like: `012345678901234567890:abcdefghijk`)

## Step 5: Edit .env File

```bash
cd ~/rick_chatbot
nano .env
```

Replace the placeholder values:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
GOOGLE_CSE_ID=your_actual_search_engine_id_here
```

## Step 6: Test the Setup

Run your chatbot - you should see:
```
🔑 Google Custom Search API configured!
```

## API Limits

- **Free tier**: 100 searches per day
- **Paid tier**: $5 per 1000 queries (after free quota)

This gives you much more reliable search results than web scraping!
EOF

# Deactivate virtual environment for final summary
deactivate 2>/dev/null || true

# Final setup summary
echo ""
echo "🎉 Enhanced Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 Project directory: $PROJECT_DIR"
echo "🐍 Virtual environment: $PROJECT_DIR/rick_env"
echo "🧠 Installed models: $MODELS_INSTALLED"
if [ -n "$DEFAULT_MODEL" ]; then
    echo "🎯 Default model: $DEFAULT_MODEL"
fi
echo ""
echo "🔑 Google Search API Setup:"
echo "   📝 Edit .env file with your Google API credentials"
echo "   📖 See GOOGLE_API_SETUP.md for detailed instructions"
echo "   🔗 Get API key: https://console.cloud.google.com/"
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
    echo "   ollama pull qwen2.5:3b-instruct  # Recommended #1"
    echo "   ollama pull phi3:mini            # Fast alternative"
    echo "   ollama pull qwen2:1.5b           # Lightweight"
    echo "   ollama pull llama3.2:1b          # Meta model"
    echo "   ollama pull gemma:2b             # Google model"
    echo ""
fi

echo "📋 Key Features:"
echo "   🔍 Internet search with Google Custom Search API"
echo "   🤖 Multiple AI backends (Ollama + transformers)"
echo "   🧠 Smart model selection (qwen2.5:3b-instruct preferred)"
echo "   💾 Persistent memory system"
echo "   🎭 Rick Sanchez personality"
echo ""
echo "🔧 To install as a system service:"
echo "   sudo cp rick-chatbot.service /etc/systemd/system/"
echo "   sudo systemctl enable rick-chatbot"
echo "   sudo systemctl start rick-chatbot"
echo ""
echo "🔍 To verify setup:"
echo "   cd $PROJECT_DIR"
echo "   source rick_env/bin/activate"
echo "   python3 -c \"import requests, colorama, dotenv; print('All packages OK')\""
echo "   ollama list  # Check models"
echo ""
echo "*burp* Wubba lubba dub dub! Your enhanced Rick chatbot is ready!"
echo "Don't forget to configure Google Search API for the best experience!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
