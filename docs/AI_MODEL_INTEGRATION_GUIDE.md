# AI Model Integration Guide for Proxima

This beginner-friendly guide explains how to add and verify AI model integration in Proxima TUI.

## Quick Start (5 Minutes)

### Step 1: Launch Proxima TUI

```bash
cd C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima
python run_tui.py
```

Or if installed:
```bash
proxima tui
```

### Step 2: Navigate to Settings

Press `5` or click on **Settings** in the footer menu.

### Step 3: Select Your AI Provider

1. Find the **"AI Assistant Settings"** section
2. Click on the **"AI Mode"** dropdown
3. Select your preferred provider from 50+ options:
   - **Local LLM (Free, Private)** - Ollama, LM Studio
   - **OpenAI API** - GPT-4, GPT-4o
   - **Anthropic API** - Claude 3.5 Sonnet
   - **Google AI** - Gemini 2.0
   - **xAI** - Grok-2
   - **DeepSeek** - DeepSeek V3, R1
   - **Mistral AI** - Mistral Large
   - **Groq** - Ultra-fast inference
   - And many more...

### Step 4: Configure Your Provider

After selecting a provider, you'll see its settings panel. Enter:
- **API Key** (for cloud providers)
- **Model Name** (select from dropdown or enter custom)
- **Endpoint URL** (for self-hosted options)

### Step 5: Test the Connection

Click the **"Verify API Key"** or **"Test Connection"** button.
- ✅ Green success message = Ready to use
- ❌ Red error message = Check your API key or connection

### Step 6: Save Settings

Click **"💾 Save Settings"** to persist your configuration.

---

## Detailed Provider Setup

### Option A: Free Local LLM (Ollama)

**Best for**: Privacy-conscious users, offline use, no API costs

1. **Install Ollama** (if not already):
   - Windows: Download from https://ollama.ai/download
   - Run the installer

2. **Download a Model**:
   ```bash
   ollama pull llama3
   ```
   Other recommended models:
   - `mistral` - Fast and capable
   - `codellama` - Great for code
   - `llama3:70b` - Most capable (needs 48GB RAM)

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Configure in Proxima**:
   - AI Mode: `Local LLM (Free, Private)`
   - Ollama URL: `http://localhost:11434`
   - Model Name: `llama3`
   - Click "Test Connection"

### Option B: OpenAI API

**Best for**: Highest quality responses, GPT-4 capabilities

1. **Get API Key**:
   - Go to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (starts with `sk-`)

2. **Configure in Proxima**:
   - AI Mode: `OpenAI API (GPT-4, GPT-4o)`
   - API Key: Paste your `sk-...` key
   - Model: Select from dropdown (GPT-4o recommended)
   - Click "Verify API Key"

3. **Cost Note**: OpenAI charges per token. GPT-4o-mini is cheapest.

### Option C: Anthropic Claude

**Best for**: Long context, nuanced responses, safety

1. **Get API Key**:
   - Go to https://console.anthropic.com/
   - Create account and get API key (starts with `sk-ant-`)

2. **Configure in Proxima**:
   - AI Mode: `Anthropic API (Claude)`
   - API Key: Paste your `sk-ant-...` key
   - Model: Claude 3.5 Sonnet (recommended)
   - Click "Verify API Key"

### Option D: Google Gemini

**Best for**: Multi-modal, free tier available

1. **Get API Key**:
   - Go to https://aistudio.google.com/apikey
   - Click "Create API Key"
   - Copy key (starts with `AIza`)

2. **Configure in Proxima**:
   - AI Mode: `Google AI (Gemini)`
   - API Key: Paste your key
   - Model: Gemini 2.0 Flash (recommended)
   - Click "Verify API Key"

### Option E: Groq (Ultra-Fast, Free Tier)

**Best for**: Speed, free usage, open models

1. **Get API Key**:
   - Go to https://console.groq.com/keys
   - Create free account
   - Generate API key (starts with `gsk_`)

2. **Configure in Proxima**:
   - AI Mode: `Groq (Fast Inference)`
   - API Key: Paste your `gsk_...` key
   - Model: Llama 3.3 70B Versatile
   - Click "Verify API Key"

### Option F: DeepSeek (Cost-Effective)

**Best for**: Reasoning tasks, very low cost

1. **Get API Key**:
   - Go to https://platform.deepseek.com/
   - Create account and get API key

2. **Configure in Proxima**:
   - AI Mode: `DeepSeek API`
   - API Key: Paste your key
   - Model: DeepSeek-V3 or DeepSeek-R1 (Reasoning)
   - Click "Verify API Key"

---

## Using AI in Proxima

### AI Thinking Panel (Execution Tab)

1. Press `2` to go to Execution tab
2. The AI Thinking Panel is on the right side
3. Type your question in the chat input
4. Click **"Send"** or press Enter
5. AI will respond with context-aware help

**What you can ask**:
- "Why is my simulation slow?"
- "Explain the current execution stage"
- "What optimization should I use for 20 qubits?"
- "Help me understand this error"

### AI Panel Controls

| Button | Action |
|--------|--------|
| **Send** | Send your message to AI |
| **🛑 Stop** | Interrupt AI thinking |
| **🗑️ Clear** | Clear conversation history |
| **💾 Export** | Save conversation to file |
| **⚙️ Settings** | Go to AI settings |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+T` | Toggle AI Thinking Panel |
| `Enter` | Send message (when in input) |

---

## Troubleshooting

### "Connection Failed" Error

1. **For local LLM (Ollama)**:
   - Check if Ollama is running: `ollama serve`
   - Verify URL is correct: `http://localhost:11434`
   - Try pulling a model: `ollama pull llama3`

2. **For cloud APIs**:
   - Check your API key is correct
   - Verify your account has credits
   - Check internet connection

### "Invalid API Key" Error

- Double-check the key format:
  - OpenAI: starts with `sk-`
  - Anthropic: starts with `sk-ant-`
  - Google: starts with `AIza`
  - Groq: starts with `gsk_`
- Ensure no extra spaces
- Generate a new key if needed

### AI Not Responding

1. Check the AI status indicator (should show "● Ready")
2. Click "🛑 Stop" if stuck on "Thinking..."
3. Try "🗑️ Clear" to reset the conversation
4. Verify your provider is configured in Settings

### Slow Responses

- **For cloud APIs**: Normal, especially for larger models
- **For local LLM**: 
  - Use a smaller model (e.g., `llama3:8b` instead of `70b`)
  - Ensure no other heavy applications are running
  - Consider using Groq for fast cloud inference

---

## Advanced: Custom API Endpoints

For self-hosted or custom OpenAI-compatible APIs:

1. Select `AI Mode: Custom API` (scroll down in dropdown)
2. Enter:
   - **API Base URL**: Your endpoint (e.g., `http://localhost:8080/v1`)
   - **API Key**: If required
   - **Model Name**: The model identifier

Compatible with:
- vLLM
- Text Generation WebUI
- LocalAI
- LM Studio (OpenAI-compatible mode)
- Any OpenAI-compatible API

---

## Provider Comparison

| Provider | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| Ollama (Local) | ⭐⭐⭐ | ⭐⭐⭐ | Free | Privacy, offline |
| Groq | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free tier | Speed |
| OpenAI GPT-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | Quality |
| Anthropic Claude | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | Long context |
| DeepSeek | ⭐⭐⭐ | ⭐⭐⭐⭐ | $ | Cost, reasoning |
| Google Gemini | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free tier | Multi-modal |

---

## Summary Checklist

- [ ] Choose a provider (start with Groq for free, fast setup)
- [ ] Get API key from provider's website
- [ ] Open Proxima Settings (press `5`)
- [ ] Select provider from AI Mode dropdown
- [ ] Enter API key
- [ ] Click "Test Connection"
- [ ] Save Settings
- [ ] Go to Execution tab (press `2`) to use AI panel
- [ ] Type a message and click Send!

---

*Need help? Press `?` in Proxima for the help screen, or check the main documentation at `docs/index.md`.*
