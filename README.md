# rick-chatbot
# 🧪 Rick Sanchez Chatbot for Raspberry Pi 5

*burp* Welcome to the most scientifically advanced chatbot in this dimension! This is an offline LLM chatbot with Rick Sanchez's personality, complete with short-term and long-term memory systems.

## 🎯 Features

- **Offline Operation**: Runs completely locally on your Raspberry Pi 5
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

### Software
- **Raspberry Pi OS (64-bit recommended)**
- Python 3.8+
- Internet connection (for initial model download)

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

3. **Start chatting:**
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
```

5. **Pre-download the model:**
```bash
python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('microsoft/DialoGPT-small'); AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')"
```

6. **Run the chatbot:**
```bash
python3 rick_chatbot.py
```

## 💻 Usage

### Basic Commands
- **Start chatbot**: `python3 rick_chatbot.py` (in activated environment)
- **Quick start**: `./start_rick.sh`
- **Exit chat**: Type `quit`, `exit`, or `bye`
- **Reset conversation**: Type `reset`
- **View memory stats**: Type `stats`

### Example Conversation
```
🧪 Rick Sanchez Chatbot v1.0
*burp* Great, another person who wants to chat.
What's your name? Morty

Rick: *burp* Morty? Fine, whatever.

Morty: Hi Rick, how are you doing?
Rick: *burp* I'm Rick Sanchez, I'm great, thanks for asking, I guess.

Morty: Can you explain quantum physics?
Rick: *burp* Finally, someone wants to talk about something interesting. Obviously quantum physics is about particles existing in multiple states until observed, *burp* but I doubt your tiny brain can handle the real complexities.

Morty: quit
Rick: *burp* Finally, some peace and quiet. Later.
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

### Model Selection
Edit `rick_chatbot.py` to change the model:
```python
# Faster but less capable
rick = RickChatbot("microsoft/DialoGPT-small")

# Better quality but slower
rick = RickChatbot("microsoft/DialoGPT-medium")
```

### Memory Settings
```python
# Adjust memory capacity
short_memory = ShortTermMemory(max_entries=30)  # Default: 20

# Change memory consolidation frequency
if self.conversation_count % 5 == 0:  # Default: 10
    self._consolidate_memories()
```

### Personality Tuning
Modify `RickPersonality.RICK_PHRASES` and `RICK_RESPONSES` to customize Rick's responses.

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce model size or add swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**2. Model Loading Slow**
- First run downloads ~500MB model
- Subsequent runs load from cache
- Consider using DialoGPT-small for faster loading

**3. Dependencies Issues**
```bash
# Reinstall in virtual environment
source rick_env/bin/activate
pip install --force-reinstall torch transformers
```

**4. Permission Errors**
```bash
# Fix file permissions
chmod +x start_rick.sh
sudo chown -R $USER:$USER ~/rick_chatbot
```

### Performance Tips

1. **Use DialoGPT-small** for fastest responses
2. **Enable GPU acceleration** (if available):
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```
3. **Increase swap space** for 4GB Pi models
4. **Close unnecessary applications** before running
5. **Use wired connection** for stability

## 📁 File Structure

```
~/rick_chatbot/
├── rick_chatbot.py          # Main chatbot code
├── rick_env/                # Python virtual environment
├── rick_memory.db           # SQLite memory database
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
pip install --upgrade transformers torch
```

### Backup Memory
```bash
cp rick_memory.db rick_memory_backup.db
```

### Reset Everything
```bash
rm -rf ~/rick_chatbot
# Then reinstall
```

## 🚨 Important Notes

- **First run takes 5-10 minutes** (model download)
- **Requires internet** for initial setup only
- **Memory usage**: ~2-4GB RAM during operation
- **Storage**: ~2GB for model and dependencies
- **Response time**: 3-10 seconds per response on Pi 5

## 🐛 Known Limitations

- Responses may be slower than cloud-based chatbots
- Limited by model size (DialoGPT constraints)
- May occasionally produce generic responses
- Memory retrieval is keyword-based (not semantic)

## 🛠️ Advanced Configuration

### Custom Personality Responses
Edit the `RICK_RESPONSES` dictionary in `rick_chatbot.py`:
```python
RICK_RESPONSES = {
    "custom_trigger": "*burp* Your custom Rick response here",
    # Add more custom responses
}
```

### Database Queries
Access the memory database directly:
```bash
sqlite3 rick_memory.db
.tables
SELECT * FROM memories WHERE importance > 8;
```

### Systemd Service (Auto-start)
```bash
sudo cp rick-chatbot.service /etc/systemd/system/
sudo systemctl enable rick-chatbot
sudo systemctl start rick-chatbot
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure adequate RAM/storage
4. Try resetting the virtual environment

## 🎓 Credits

- **Rick and Morty**: Created by Dan Harmon and Justin Roiland
- **DialoGPT**: Microsoft Research
- **Transformers**: Hugging Face
- **PyTorch**: Facebook AI Research

---

*burp* Remember, existence is pain, but at least now you have a chatbot that understands quantum mechanics!

**Wubba lubba dub dub!** 🧪🚀
