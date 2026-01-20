# Proxima: Complete Beginner's Guide

## 🎉 Welcome to Proxima!

**Don't worry if you're new to programming or quantum computing!** This guide is written specifically for absolute beginners. We'll walk you through every single step with clear explanations.

---

## 📖 Table of Contents

1. [What is Proxima?](#what-is-proxima)
2. [What You'll Need](#what-youll-need)
3. [Step 1: Installing Python](#step-1-installing-python)
4. [Step 2: Installing Proxima](#step-2-installing-proxima)
5. [Step 3: Verifying Your Installation](#step-3-verifying-your-installation)
6. [Step 4: Running Your First Simulation](#step-4-running-your-first-simulation)
7. [Step 5: Understanding the Results](#step-5-understanding-the-results)
8. [Common Tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)
10. [Getting Help](#getting-help)

---

## What is Proxima?

**Proxima** is a tool that lets you run quantum computer simulations on your regular computer. Think of it as a "practice mode" for quantum computing - you can experiment with quantum circuits without needing access to an actual quantum computer.

### Why Would I Use This?

- **Learning**: Understand how quantum computers work
- **Testing**: Try out quantum algorithms before running them on real hardware
- **Research**: Experiment with quantum circuits quickly and easily
- **Education**: Great for students and teachers

### What Can Proxima Do?

- ✅ Run quantum circuit simulations
- ✅ Compare results across different simulation methods
- ✅ Benchmark performance
- ✅ Export results in various formats
- ✅ Use AI to help interpret results (optional)

---

## What You'll Need

Before we start, let's make sure you have everything:

| Item | Description | Do You Have It? |
|------|-------------|-----------------|
| **A Computer** | Windows, Mac, or Linux | ✅ You're reading this! |
| **Internet Connection** | To download software | ✅ |
| **15-30 Minutes** | Time to complete setup | ⏰ |
| **Python** | Programming language (we'll install this) | 📥 We'll help you |

**You do NOT need:**
- ❌ Programming experience
- ❌ Knowledge of quantum physics
- ❌ A powerful computer
- ❌ Any special hardware

---

## Step 1: Installing Python

Python is a programming language that Proxima needs to run. Don't worry - you won't need to learn programming!

### 🪟 For Windows Users

#### Step 1.1: Download Python

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)

2. **Go to the Python website:**
   ```
   https://www.python.org/downloads/
   ```

3. **Click the big yellow button** that says "Download Python 3.x.x"
   
   ![Download button is usually at the top of the page]

4. **Wait for the download** to complete (the file will be in your Downloads folder)

#### Step 1.2: Install Python

1. **Find the downloaded file** - it will be named something like `python-3.12.0-amd64.exe`

2. **Double-click the file** to run it

3. **⚠️ IMPORTANT:** Check the box that says:
   ```
   ☑️ Add Python to PATH
   ```
   This is at the bottom of the installer window. **This step is crucial!**

4. **Click "Install Now"**

5. **Wait for installation** to complete (this takes 1-5 minutes)

6. **Click "Close"** when you see "Setup was successful"

#### Step 1.3: Verify Python Installation

1. **Press `Windows Key + R`** on your keyboard

2. **Type `cmd`** and press Enter

3. **A black window will open** (this is called Command Prompt)

4. **Type this exactly and press Enter:**
   ```
   python --version
   ```

5. **You should see something like:**
   ```
   Python 3.12.0
   ```

   **If you see this, congratulations! Python is installed.** 🎉

   **If you see an error**, close Command Prompt and restart your computer, then try again.

---

### 🍎 For Mac Users

#### Step 1.1: Download Python

1. **Open Safari or your preferred browser**

2. **Go to:**
   ```
   https://www.python.org/downloads/
   ```

3. **Click "Download Python 3.x.x"**

4. **Wait for download** to complete

#### Step 1.2: Install Python

1. **Open Finder** and go to your **Downloads** folder

2. **Double-click** the file (named something like `python-3.12.0-macos11.pkg`)

3. **Click "Continue"** through the installer screens

4. **Click "Install"** (you may need to enter your Mac password)

5. **Wait for installation** to complete

6. **Click "Close"**

#### Step 1.3: Verify Python Installation

1. **Press `Command + Space`** to open Spotlight

2. **Type `Terminal`** and press Enter

3. **A window will open** with a command line

4. **Type this exactly and press Enter:**
   ```
   python3 --version
   ```

5. **You should see something like:**
   ```
   Python 3.12.0
   ```

---

### 🐧 For Linux Users

Most Linux systems come with Python pre-installed. Open a terminal and check:

```bash
python3 --version
```

If Python isn't installed, use your package manager:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip
```

---

## Step 2: Installing Proxima

Now that Python is installed, let's install Proxima!

### 🪟 Windows Instructions

1. **Press `Windows Key + R`**

2. **Type `cmd`** and press Enter

3. **In the black Command Prompt window, type this command exactly:**
   ```
   pip install proxima-agent
   ```

4. **Press Enter** and wait

5. **You'll see text scrolling** as packages are downloaded and installed

6. **When you see the prompt again** (blinking cursor), installation is complete!

### 🍎 Mac Instructions

1. **Open Terminal** (Command + Space, type "Terminal")

2. **Type this command:**
   ```
   pip3 install proxima-agent
   ```

3. **Press Enter** and wait for installation to complete

### 🐧 Linux Instructions

1. **Open your terminal**

2. **Type this command:**
   ```
   pip3 install proxima-agent
   ```

3. **Press Enter** and wait for installation

---

### Installing Extra Features (Optional)

If you want additional features, you can install them too:

**For AI-powered insights:**
```
pip install proxima-agent[llm]
```

**For a fancy terminal interface:**
```
pip install proxima-agent[ui]
```

**For everything:**
```
pip install proxima-agent[all]
```

---

## Step 3: Verifying Your Installation

Let's make sure Proxima is working!

### Check Version

1. **Open your terminal/Command Prompt** (same as before)

2. **Type:**
   ```
   proxima --version
   ```

3. **You should see something like:**
   ```
   Proxima v0.3.0
   ```

### Check Available Backends

**Type:**
```
proxima backends list
```

**You should see a list of available simulation backends, like:**
```
Available Backends:
  ✓ lret (default)
  ✓ cirq
  ✓ qiskit
  ○ quest (not installed)
  ○ cuquantum (requires GPU)
```

**✓ means available, ○ means not installed (that's okay!)**

### Get Help

**Type:**
```
proxima --help
```

**This shows all available commands.**

---

## Step 4: Running Your First Simulation

Now for the exciting part - let's run a quantum simulation!

### What is a Bell State?

A "Bell State" is one of the simplest quantum circuits. It creates a special quantum connection between two quantum bits (qubits). Don't worry about the physics - just know it's a great first experiment!

### Method 1: Quick Run (Easiest)

**Type this command:**
```
proxima run "bell state"
```

**Press Enter and wait a few seconds...**

**You should see output like:**
```
╭─ Execution Summary ─────────────────────────────────╮
│ Circuit: Bell State                                  │
│ Backend: lret                                        │
│ Qubits: 2                                           │
│ Shots: 1000                                         │
│ Time: 0.023s                                        │
╰─────────────────────────────────────────────────────╯

Results:
  |00⟩: 498 (49.8%)
  |11⟩: 502 (50.2%)

Status: ✓ Success
```

**Congratulations! You just ran your first quantum simulation!** 🎉

---

### Method 2: Create a Circuit File

For more control, you can create your own circuit files.

#### Step 4.1: Create a Project Folder

**Windows:**
1. Open File Explorer
2. Go to your Documents folder
3. Right-click → New → Folder
4. Name it `my-quantum-project`

**Mac/Linux:**
```
mkdir ~/my-quantum-project
cd ~/my-quantum-project
```

#### Step 4.2: Initialize Proxima

**Navigate to your folder in terminal:**

**Windows:**
```
cd Documents\my-quantum-project
```

**Mac/Linux:**
```
cd ~/my-quantum-project
```

**Then run:**
```
proxima init
```

**This creates a project structure with example files!**

#### Step 4.3: Run the Example Circuit

**Type:**
```
proxima run circuits/examples/bell.json
```

---

### Method 3: Natural Language (Easy Mode)

Proxima understands plain English! Try these:

```
proxima run "Create a 3-qubit GHZ state"
```

```
proxima run "Simple quantum teleportation"
```

```
proxima run "2 qubit superposition"
```

---

## Step 5: Understanding the Results

Let's decode what the output means:

### Example Output Explained

```
Results:
  |00⟩: 498 (49.8%)
  |11⟩: 502 (50.2%)
```

| Symbol | What It Means |
|--------|---------------|
| `|00⟩` | Both qubits measured as 0 |
| `|11⟩` | Both qubits measured as 1 |
| `498` | Number of times we got this result |
| `49.8%` | Percentage of total measurements |

### Why These Numbers?

In a Bell state:
- The qubits are "entangled" (connected in a special quantum way)
- When you measure them, they're always the same (both 0 or both 1)
- The split is roughly 50/50 because quantum mechanics is probabilistic

**This is exactly what quantum theory predicts!** 🔬

---

## Common Tasks

Here are things you'll do often:

### Run a Simulation

```
proxima run "bell state"
proxima run circuits/my_circuit.json
proxima run circuit.qasm --backend cirq
```

### Change Number of Shots

**More shots = more accurate results:**
```
proxima run "bell state" --shots 10000
```

### Use a Different Backend

```
proxima run "bell state" --backend cirq
proxima run "bell state" --backend qiskit
```

### Compare Multiple Backends

```
proxima compare "bell state" --backends lret,cirq,qiskit
```

### See Your Previous Results

```
proxima results list
```

### Export Results

**Save as JSON:**
```
proxima export last --format json --output my_results.json
```

**Save as CSV:**
```
proxima export last --format csv --output my_results.csv
```

### Run Benchmarks

```
proxima benchmark run "bell state" --runs 10
```

### View Configuration

```
proxima config show
```

### Change Default Backend

```
proxima config set backends.default cirq
```

### Launch Interactive Terminal UI

```
proxima tui
```

---

## Troubleshooting

### ❌ "proxima: command not found" or "'proxima' is not recognized"

**Cause:** Python's scripts folder isn't in your system PATH.

**Fix (Windows):**
1. Close Command Prompt
2. Search for "Environment Variables" in Start Menu
3. Click "Edit the system environment variables"
4. Click "Environment Variables" button
5. Under "User variables", find "Path" and click "Edit"
6. Click "New" and add:
   ```
   %APPDATA%\Python\Python312\Scripts
   ```
   (Replace `Python312` with your version)
7. Click OK on all windows
8. Open a new Command Prompt and try again

**Alternative Fix:**
```
python -m proxima --version
```
Use `python -m proxima` instead of just `proxima`

---

### ❌ "pip: command not found"

**Fix (Windows):**
```
python -m pip install proxima-agent
```

**Fix (Mac/Linux):**
```
python3 -m pip install proxima-agent
```

---

### ❌ "Permission denied" error

**Fix (Mac/Linux):**
```
pip3 install --user proxima-agent
```

---

### ❌ Installation is very slow

This is normal for the first installation. Just wait patiently. If it's been more than 10 minutes:
1. Press `Ctrl + C` to cancel
2. Try again with:
   ```
   pip install proxima-agent --timeout 120
   ```

---

### ❌ "No module named 'proxima'"

**Fix:**
```
pip install proxima-agent --upgrade
```

---

### ❌ Results look wrong or unexpected

**Try more shots for accuracy:**
```
proxima run "bell state" --shots 10000
```

---

### ❌ "Backend not available" error

Some backends require additional packages:

**For Cirq:**
```
pip install cirq
```

**For Qiskit:**
```
pip install qiskit qiskit-aer
```

---

## Getting Help

### Built-in Help

**General help:**
```
proxima --help
```

**Help for a specific command:**
```
proxima run --help
proxima benchmark --help
proxima backends --help
```

### Online Resources

- **Documentation**: https://proxima-project.github.io/proxima/
- **GitHub Issues**: https://github.com/prthmmkhija1/Pseudo-Proxima/issues
- **Discussions**: https://github.com/prthmmkhija1/Pseudo-Proxima/discussions

### Community

Having trouble? You can:
1. Check existing GitHub issues for similar problems
2. Open a new issue with details about your problem
3. Include your operating system and Python version

---

## What's Next?

Now that you've got Proxima running, here are some next steps:

### 📚 Learn More About Quantum Computing

- [IBM Quantum Learning](https://learning.quantum.ibm.com/)
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [Quantum Country](https://quantum.country/)

### 🧪 Try More Complex Circuits

```
proxima run "GHZ state with 4 qubits"
proxima run "quantum teleportation"
proxima run "simple VQE ansatz"
```

### 📊 Explore Benchmarking

```
proxima benchmark suite quick
proxima benchmark compare "bell state" --backends all
```

### 🎨 Use the Terminal UI

```
proxima tui
```

### 🤖 Enable AI Insights (Optional)

If you have an OpenAI API key:
```
proxima config set llm.provider openai
proxima config set llm.api_key YOUR_API_KEY
proxima run "bell state" --explain
```

---

## Quick Reference Card

| What You Want to Do | Command |
|---------------------|---------|
| Check version | `proxima --version` |
| Get help | `proxima --help` |
| Run a simulation | `proxima run "bell state"` |
| Run from file | `proxima run circuit.json` |
| More shots | `proxima run "bell state" --shots 10000` |
| Different backend | `proxima run "bell state" --backend cirq` |
| Compare backends | `proxima compare "bell state"` |
| See results | `proxima results list` |
| Export results | `proxima export last --format json` |
| List backends | `proxima backends list` |
| View config | `proxima config show` |
| Interactive mode | `proxima tui` |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Qubit** | Quantum bit - the basic unit of quantum information |
| **Circuit** | A sequence of quantum operations |
| **Gate** | An operation on qubits (like H, CNOT, X, Z) |
| **Shots** | Number of times to run the simulation |
| **Backend** | The simulation engine used |
| **Bell State** | A simple 2-qubit entangled state |
| **GHZ State** | A multi-qubit entangled state |
| **Superposition** | Qubit being in multiple states at once |
| **Entanglement** | Special connection between qubits |
| **Measurement** | Reading the final state of qubits |

---

## Congratulations! 🎉

You've successfully:
- ✅ Installed Python
- ✅ Installed Proxima
- ✅ Run your first quantum simulation
- ✅ Learned the basic commands

**You're now ready to explore the quantum world!**

---

*This guide was written for Proxima v0.3.0. If you're using a different version, some details may vary.*

*Last updated: January 2026*
