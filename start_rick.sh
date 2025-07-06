#!/bin/bash
# Rick Sanchez Chatbot Startup Script - Enhanced Edition
# *burp* Let's get this interdimensional chat with internet search started!

echo "🧪 Starting Rick Sanchez Chatbot - Enhanced Edition..."
echo "*burp* Initializing quantum neural pathways and search algorithms..."

# Navigate to the rick_chatbot directory
RICK_DIR="$HOME/rick_chatbot"

if [ ! -d "$RICK_DIR" ]; then
    echo "❌ Error: Rick chatbot directory not found at $RICK_DIR"
    echo "Please run the setup script first!"
    exit 1
fi

cd "$RICK_DIR"

# Check if virtual environment exists
if [ ! -d "rick_env" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please run the setup script first!"
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

# Check if main script exists
if [ ! -f "rick_chatbot.py" ]; then
    echo "❌ Error: rick_chatbot.py not found!"
    echo "Please make sure you've downloaded the enhanced chatbot script."
    exit 1
fi

# Check if Ollama is running, start if needed
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🚀 Starting Ollama server..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
    echo "✅ Ollama server started"
else
    echo "✅ Ollama server already running"
fi

# Check available models
echo "🧠 Checking available AI models..."
if command -v ollama &> /dev/null; then
    MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)
    if [ "$MODELS" -gt 0 ]; then
        echo "✅ Found $MODELS Ollama models"
        # Show preferred model if available
        if ollama list 2>/dev/null | grep -q "qwen2.5:3b-instruct"; then
            echo "🎯 qwen2.5:3b-instruct available (recommended)"
        elif ollama list 2>/dev/null | grep -q "phi3:mini"; then
            echo "🔧 phi3:mini available (fallback)"
        fi
    else
        echo "⚠️  No Ollama models found. Install with:"
        echo "   ollama pull qwen2.5:3b-instruct"
    fi
else
    echo "⚠️  Ollama not found - using traditional transformers"
fi

# Check for Google API configuration
if [ -f ".env" ]; then
    if grep -q "your_google_api_key_here" .env 2>/dev/null; then
        echo "⚠️  Google Search API not configured"
        echo "📋 Edit .env file to add your Google API credentials for better search results"
        echo "🔗 See GOOGLE_API_SETUP.md for instructions"
    elif grep -q "GOOGLE_API_KEY=" .env 2>/dev/null && ! grep -q "your_google_api_key_here" .env; then
        echo "🔑 Google Search API configuration detected"
        echo "🌐 Internet search will use Google Custom Search API"
    else
        echo "📝 .env file exists but may need Google API configuration"
    fi
else
    echo "⚠️  No .env file found - web search will use fallback methods"
    echo "💡 Create .env file with Google API credentials for better search"
fi

# Check internet connectivity
echo "🌐 Checking internet connectivity..."
if ping -c 1 google.com &> /dev/null; then
    echo "✅ Internet connection available - search enabled"
elif ping -c 1 8.8.8.8 &> /dev/null; then
    echo "✅ Internet connection available - search enabled"
else
    echo "⚠️  No internet connection - search disabled"
fi

# Set environment variables for better Pi performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Start the chatbot
echo "🚀 Starting Rick Sanchez Enhanced Edition..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 rick_chatbot.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "*burp* Chat session ended. Wubba lubba dub dub!"
echo ""
echo "💡 Tips for next time:"
echo "   • Configure Google API in .env for better search"
echo "   • Try 'debug' mode to see what's happening"
echo "   • Use 'search <query>' to force web searches"
echo "   • Type 'models' to see available AI models"
