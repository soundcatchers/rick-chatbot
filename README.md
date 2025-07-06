# rick-chatbot
# 🧪 Rick Sanchez Chatbot for Raspberry Pi 5 - Enhanced Edition

*burp* Welcome to the most scientifically advanced chatbot in this dimension! This is an offline LLM chatbot with Rick Sanchez's personality, complete with internet search capabilities and multiple AI backend support.

## 🎯 Features

- **Multiple AI Backends**: Supports both Ollama (recommended) and traditional transformers
- **Internet Search Integration**: Real-time web search with Google Custom Search API
- **Rick Sanchez Personality**: Cynical, brilliant, and burp-filled responses
- **Dual Memory System**: 
  - Short-term memory for recent conversations
  - Long-term SQLite database for persistent memories
- **Optimized for Pi 5**: CPU-optimized model loading and inference
- **Virtual Environment**: Isolated Python environment for clean installation

## 📋 Requirements

### Hardware
- **Raspberry Pi 5 (8GB RAM recommended)**
- MicroSD card (32GB+ recommended)
- Stable power supply
- Keyboard and display (for initial setup)
- **Internet connection** (for search functionality)

### Software
- **Raspberry Pi OS (64-bit recommended)**
- Python 3.8+
- Internet connection (for initial model download and search)

## 🚀 Quick Installation

### Method 1: Automated Setup (Recommended)

1. **Download the setup script:**
```bash
wget https://raw.githubusercontent.com/soundcatchers/rick-chatbot/main/setup_pi.sh
chmod +x setup_pi.sh
```

2. **Run the setup script:**
```bash
./setup_pi.sh
```

3. **Configure Google Search (Optional but Recommended):**
```bash
cd ~/rick_chatbot
nano .env
```
Add your Google API credentials (see [Google Search Setup](#-google-search-api-setup) below)

4. **Start chatting:**
```bash
cd ~/rick_chatbot
./start_rick.sh
```

### Method 2: Manual Installation

1. **Clone or download the files:**
```bash
mkdir ~/rick_chatbot
cd ~/rick_chatbot
# Copy rick_chatbot.py to this directory
```

2. **Update system packages:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv build-essential -y
```

3. **Create virtual environment:**
```bash
python3 -m venv rick_env
source rick_env/bin/activate
```

4. **Install dependencies:**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers datasets accelerate numpy scipy
pip install requests colorama beautifulsoup4 python-dotenv
```

5. **Install and configure Ollama (Recommended):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:3b-instruct  # Recommended model
ollama pull phi3:mini           # Alternative model
```

6. **Configure Google Search API (Optional):**
```bash
nano .env
# Add your Google API key and Custom Search Engine ID
```

7. **Run the chatbot:**
```bash
python3 rick_chatbot.py
```

## 🔍 Google Search API Setup

For the best search results, configure Google Custom Search API:

### Step 1: Install python-dotenv
```bash
pip install python-dotenv
```

### Step 2: Get Google API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Go to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **API Key**
5. Copy the API key

### Step 3: Enable Custom Search API
1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Custom Search API"
3. Click on it and press **Enable**

### Step 4: Create Custom Search Engine
1. Go to [Google Custom Search Engine](https://cse.google.com/cse/)
2. Click **Add** to create a new search engine
3. For "Sites to search", enter `*` (to search the entire web)
4. Give it a name like "Rick Chatbot Search"
5. Click **Create**
6. Copy the **Search Engine ID**

### Step 5: Create .env File
```bash
cd ~/rick_chatbot
nano .env
```

Add your credentials:
```bash
# Google Custom Search API Configuration
GOOGLE_API_KEY=your_actual_api_key_here
GOOGLE_CSE_ID=your_actual_search_engine_id_here
```

### API Limits
- **Free tier**: 100 searches per day
- **Paid tier**: $5 per 1000 queries (after free quota)

## 💻 Usage

### Basic Commands
- **Start chatbot**: `./start_rick.sh` or `python3 rick_chatbot.py` (in activated environment)
- **Exit chat**: Type `quit`, `exit`, or `bye`
- **Force web search**: `search <query>`
- **Toggle debug mode**: `debug`
- **View available models**: `models`
- **Reset conversation**: Type `reset`
- **View memory stats**: Type `stats`

### Example Conversation
```
🧪 Rick Sanchez Chatbot - Enhanced Edition with Internet Search
*burp* Loading advanced AI systems...
🌐 Internet connection detected - search enabled!
🔑 Google Custom Search API configured!
✅ Connected to Ollama!
🤖 Using model: qwen2.5:3b-instruct

*burp* Welcome to Rick's interdimensional chat experience!

You: who is the current uk prime minister
*burp* Searching the interdimensional web for: who is the current uk prime minister...
*burp* Found interdimensional knowledge from Google API!
Rick: *burp* Look, the current UK Prime Minister is Keir Starmer. He's been in charge since July 2024 when Labour won the election. *burp* Not that it matters much in the grand scheme of the multiverse, but there you go.

You: explain quantum physics
Rick: *burp* Finally, someone wants to talk about something interesting! Obviously quantum physics is about particles existing in multiple states until observed, but I doubt your tiny brain can handle the real complexities like quantum entanglement and the many-worlds interpretation...

You: quit
Rick: *burp* Finally! I've got science to do. Peace out!
```

## 🧠 Memory System

### Short-Term Memory
- Stores last 20 conversation entries
- Maintains conversation context
- Includes personality traits and user preferences

### Long-Term Memory
- SQLite database (`rick_memory.db`)
- Stores important conversations (importance ≥ 8)
- Retrieves relevant memories based on keywords
- Persists between sessions

## ⚙️ Configuration

### Model Priority (Ollama)
The chatbot automatically selects the best available model in this order:
1. **qwen2.5:3b-instruct** (Recommended - fast and intelligent)
2. **qwen2.5:3b** (Alternative)
3. **phi3:mini** (Microsoft Phi-3)
4. **qwen2:1.5b** (Qwen 1.5B)
5. **llama3.2:1b** (Meta Llama)
6. **gemma:2b** (Google Gemma)

### Install Additional Models
```bash
ollama pull qwen2.5:3b-instruct  # Recommended
ollama pull phi3:mini           # Fast alternative
ollama pull llama3.2:3b         # Larger, more capable
```

### Search Engine Priority
The chatbot tries search engines in this order:
1. **Google Custom Search API** (if configured)
2. **Google web scraping** (fallback)
3. **Bing web scraping**
4. **Wikipedia API**
5. **DuckDuckGo API**

### Memory Settings
```python
# Adjust memory capacity
short_memory = ShortTermMemory(max_entries=30)  # Default: 20

# Change memory consolidation frequency
if self.conversation_count % 5 == 0:  # Default: 10
    self._consolidate_memories()
```

## 🔧 Troubleshooting

### Common Issues

**1. Google Search Not Working**
```bash
# Check if .env file exists and has correct format
cat .env
# Should show:
# GOOGLE_API_KEY=your_key_here
# GOOGLE_CSE_ID=your_cse_id_here

# Test API key
curl "https://www.googleapis.com/customsearch/v1?key=YOUR_API_KEY&cx=YOUR_CSE_ID&q=test"
```

**2. Model Not Found**
```bash
# List available models
ollama list

# Pull recommended model
ollama pull qwen2.5:3b-instruct

# Check if Ollama is running
pgrep ollama
```

**3. Out of Memory Error**
```bash
# Reduce model size or add swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**4. Virtual Environment Issues**
```bash
# Reinstall virtual environment
rm -rf ~/rick_chatbot/rick_env
cd ~/rick_chatbot
python3 -m venv rick_env
source rick_env/bin/activate
pip install -r requirements.txt
```

**5. Search Returns Wrong Results**
- Verify Google API is configured correctly
- Check debug mode: type `debug` then try your search
- Try forcing a search: `search your query here`

### Performance Tips

1. **Use qwen2.5:3b-instruct** for best balance of speed and capability
2. **Configure Google Search API** for accurate search results
3. **Increase swap space** for 4GB Pi models
4. **Close unnecessary applications** before running
5. **Use wired connection** for search stability

## 📁 File Structure

```
~/rick_chatbot/
├── rick_chatbot.py          # Main chatbot code (Enhanced Edition)
├── rick_env/                # Python virtual environment
├── rick_memory.db           # SQLite memory database
├── .env                     # Google API configuration (create this)
├── start_rick.sh            # Quick start script
├── requirements.txt         # Python dependencies
├── setup_pi.sh             # Installation script
└── README.md               # This file
```

## 🔄 Updates and Maintenance

### Update the Chatbot
```bash
cd ~/rick_chatbot
source rick_env/bin/activate
pip install --upgrade transformers torch requests colorama python-dotenv
ollama pull qwen2.5:3b-instruct  # Update model
```

### Backup Memory and Configuration
```bash
cp rick_memory.db rick_memory_backup.db
cp .env .env.backup
```

### Reset Everything
```bash
rm -rf ~/rick_chatbot
# Then reinstall with setup script
```

## 🚨 Important Notes

- **First run takes 5-10 minutes** (model download)
- **Requires internet** for search functionality and initial setup
- **Memory usage**: ~3-6GB RAM during operation with qwen2.5:3b-instruct
- **Storage**: ~4GB for models and dependencies
- **Response time**: 2-8 seconds per response on Pi 5
- **Search quota**: 100 free Google searches per day

## 🐛 Known Limitations

- Responses may be slower than cloud-based chatbots
- Google Search API has daily quotas (100 free searches)
- May occasionally produce generic responses
- Memory retrieval is keyword-based (not semantic)
- Web scraping fallbacks may be unreliable

## 🛠️ Advanced Configuration

### Custom Model Priority
Edit `rick_chatbot.py` to change model preferences:
```python
preferred_models = [
    "your-custom-model",
    "qwen2.5:3b-instruct",
    # ... rest of the list
]
```

### Search Engine Configuration
Disable specific search engines by commenting them out:
```python
search_methods = [
    ("Google API", self.search_google_api),
    # ("Bing", self.search_bing_api),  # Disabled
    ("Wikipedia", self.search_wikipedia_api),
    ("DuckDuckGo", self.search_duckduckgo)
]
```

### Environment Variables
Additional configuration options in `.env`:
```bash
# Google Search API
GOOGLE_API_KEY=your_key_here
GOOGLE_CSE_ID=your_cse_id_here

# Optional: Custom settings
OLLAMA_HOST=http://localhost:11434
MAX_SEARCH_RESULTS=5
DEBUG_MODE=false
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Test with `debug` mode enabled
4. Check Ollama is running: `ollama list`
5. Verify internet connectivity for search
6. Ensure adequate RAM/storage

Common commands for debugging:
```bash
# Check virtual environment
source rick_env/bin/activate && which python3

# Test Ollama
ollama list
curl http://localhost:11434/api/tags

# Test Google API
curl "https://www.googleapis.com/customsearch/v1?key=YOUR_KEY&cx=YOUR_CSE&q=test"

# Check logs
tail -f ollama.log
```

## 🎓 Credits

- **Rick and Morty**: Created by Dan Harmon and Justin Roiland
- **Ollama**: Local AI model runner
- **Qwen2.5**: Alibaba Cloud's language model
- **DialoGPT**: Microsoft Research
- **Transformers**: Hugging Face
- **PyTorch**: Meta AI Research
- **Google Custom Search**: Google LLC

---

*burp* Remember, existence is pain, but at least now you have a chatbot that can search the interdimensional web AND understands quantum mechanics!

**Wubba lubba dub dub!** 🧪🚀🔍
