#!/bin/bash
# Rick Sanchez Chatbot Startup Script
# *burp* Let's get this interdimensional chat started!

echo "🧪 Starting Rick Sanchez Chatbot..."
echo "*burp* Initializing quantum neural pathways..."

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

# Check if main script exists
if [ ! -f "rick_chatbot.py" ]; then
    echo "❌ Error: rick_chatbot.py not found!"
    echo "Please make sure you've downloaded the main chatbot script."
    exit 1
fi

# Set environment variables for better Pi performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Start the chatbot
echo "🚀 Starting Rick Sanchez..."
python3 rick_chatbot.py

echo "*burp* Chat session ended. Wubba lubba dub dub!"
