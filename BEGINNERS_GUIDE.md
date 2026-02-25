# Proxima Beginner's Guide

**Complete Guide for Absolute Beginners**  
*Last Updated: January 2026*

---

## 📋 Table of Contents

1. [What is Proxima?](#what-is-proxima)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Your First Steps](#your-first-steps)
5. [Understanding Proxima Commands](#understanding-proxima-commands)
6. [Running Quantum Simulations](#running-quantum-simulations)
7. [Using the Terminal UI (TUI)](#using-the-terminal-ui-tui)
8. [Understanding Backends](#understanding-backends)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)
11. [Quick Reference](#quick-reference)

---

## What is Proxima?

### In Simple Terms

**Proxima** is a tool that helps you run quantum computer simulations on your regular computer. Think of it as:

- A **translator** that works with different quantum simulators
- A **manager** that picks the best simulator for your task
- A **controller** that lets you run quantum programs easily

### What Can You Do With It?

✅ Run quantum circuits (quantum programs)  
✅ Compare different simulation methods  
✅ Get insights about your quantum programs  
✅ Use a visual interface (TUI) or command line  
✅ Manage and view your simulation results

### Who Is This For?

- **Researchers** testing quantum algorithms
- **Students** learning quantum computing
- **Developers** building quantum applications
- **Anyone** curious about quantum simulations

---

## Prerequisites

### What You Need

#### 1. **Python** (Required)
- **Version**: Python 3.11 or newer
- **Check if you have it**: Open PowerShell and type:
  ```powershell
  python --version
  ```
- **If you don't have it**: Download from [python.org](https://www.python.org/downloads/)
  - ⚠️ **IMPORTANT**: When installing, check the box "Add Python to PATH"

#### 2. **PowerShell** (Already on Windows)
- You're already using it! It's the blue terminal window.

#### 3. **Internet Connection** (For installation)
- Needed to download packages

#### 4. **About 2 GB Free Disk Space**

---

## Installation

### Step 1: Open PowerShell

**How to open PowerShell:**
1. Press `Windows Key + X`
2. Click "Windows PowerShell" or "Terminal"

OR

1. Click Start Menu
2. Type "PowerShell"
3. Click "Windows PowerShell"

### Step 2: Navigate to Proxima Directory

Type this command (copy and paste it — replace `C:\Proxima` with wherever you cloned the project):
```powershell
cd C:\Proxima
```

Press `Enter`.

### Step 3: Create a Virtual Environment

**What's a virtual environment?** It's like a separate box for Proxima's files so they don't interfere with other programs.

```powershell
python -m venv venv
```

This creates a folder called `venv` in your directory.

### Step 4: Activate the Virtual Environment

```powershell
.\venv\Scripts\activate
```

✅ **Success!** You should see `(venv)` at the start of your command line:
```
(venv) PS C:\Proxima>
```

### Step 5: Install Proxima

**Option A: Install Everything (Recommended for beginners)**
```powershell
pip install -e .[all]
```

**Option B: Install Basic Version**
```powershell
pip install -e .
```

**Option C: Install with TUI Only**
```powershell
pip install -e .[ui]
```

⏳ **Wait**: This might take 5-10 minutes. You'll see lots of text scrolling by - that's normal!

### Step 6: Verify Installation

```powershell
proxima --version
```

✅ **You should see**: `Proxima version 0.1.0`

🎉 **Congratulations!** Proxima is now installed!

---

## Your First Steps

### 1. Check That Everything Works

Run the diagnostics command:
```powershell
proxima doctor
```

This checks:
- ✓ Python version
- ✓ Configuration files
- ✓ Available backends
- ✓ Data storage

### 2. View Available Commands

```powershell
proxima --help
```

This shows all the things you can do with Proxima.

### 3. Initialize Configuration

```powershell
proxima init
```

This creates your personal settings file.

### 4. Check Current Status

```powershell
proxima status
```

This shows:
- What backend (simulator) you're using
- Recent simulations
- Current settings

---

## Understanding Proxima Commands

### Command Structure

All Proxima commands follow this pattern:
```
proxima [COMMAND] [OPTIONS] [ARGUMENTS]
```

### Main Commands

| Command | What It Does | Example |
|---------|-------------|---------|
| `run` | Run a quantum simulation | `proxima run "bell state"` |
| `backends` | List available simulators | `proxima backends list` |
| `config` | View/change settings | `proxima config show` |
| `compare` | Compare different simulators | `proxima compare "bell state" --all` |
| `history` | View past simulations | `proxima history list` |
| `ui` | Launch visual interface | `proxima ui` |
| `status` | Check current status | `proxima status` |
| `doctor` | Run health checks | `proxima doctor` |

### Command Shortcuts (Aliases)

Proxima has shortcuts to save typing:

| Shortcut | Full Command | Example |
|----------|-------------|---------|
| `r` | `run` | `proxima r "bell state"` |
| `be` | `backends list` | `proxima be` |
| `cfg` | `config show` | `proxima cfg` |
| `hist` | `history list` | `proxima hist` |
| `cmp` | `compare` | `proxima cmp "bell state"` |

**View all shortcuts:**
```powershell
proxima aliases
```

---

## Running Quantum Simulations

### What is a Quantum Simulation?

A quantum simulation runs a quantum circuit (program) and shows you the results. Think of it like:
- **Input**: A quantum circuit (instructions)
- **Process**: The simulator runs it
- **Output**: Measurement results and statistics

### Your First Simulation: Bell State

A Bell state is a simple quantum program that creates "entanglement" between two qubits.

```powershell
proxima run "create bell state"
```

**What happens:**
1. Proxima creates a Bell state circuit
2. Picks the best simulator automatically
3. Runs the simulation
4. Shows you the results

**Expected output:**
```
✓ Backend selected: lret (reason: Fast for 2 qubits)
✓ Executing circuit...
✓ Results:
   |00⟩: 50%
   |11⟩: 50%
✓ Execution time: 0.02s
```

### More Circuit Examples

#### 1. Bell State (2 qubits)
```powershell
proxima run "bell state"
# OR use the shortcut
proxima bell
```

#### 2. GHZ State (3+ qubits)
```powershell
proxima run "create ghz state with 3 qubits"
# OR use the shortcut
proxima ghz --qubits 3
```

#### 3. Quantum Fourier Transform
```powershell
proxima run "quantum fourier transform on 4 qubits"
# OR use the shortcut
proxima qft --qubits 4
```

### Specifying Options

#### Choose a Specific Backend
```powershell
proxima run "bell state" --backend cirq
```

#### Change Number of Measurements (Shots)
```powershell
proxima run "bell state" --shots 2000
```

#### Combine Options
```powershell
proxima run "bell state" --backend qiskit --shots 5000
```

### Understanding Results

After running a simulation, you'll see:

1. **Backend Used**: Which simulator ran it (e.g., `lret`, `cirq`, `qiskit`)
2. **Measurement Results**: Probability distribution
   - Example: `|00⟩: 50%` means "50% chance of measuring 00"
3. **Execution Time**: How long it took
4. **Additional Insights**: Analysis and recommendations (if LLM is enabled)

---

## Using the Terminal UI (TUI)

### What is the TUI?

The **Terminal UI** is a visual interface that runs in your terminal. It's easier than typing commands because:
- You can click on options
- See everything at once
- Navigate with arrow keys
- More beginner-friendly

### Starting the TUI

**Method 1: Using the launcher script**
```powershell
python run_tui.py
```

**Method 2: Using the command**
```powershell
proxima ui
```

### TUI Navigation

Once the TUI opens:

#### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `1` | Go to Dashboard |
| `2` | Go to Execution (run simulations) |
| `3` | Go to Results (view history) |
| `4` | Go to Backends (simulators) |
| `5` | Go to Settings |
| `Ctrl+P` | Open command palette |
| `?` | Show help |
| `Ctrl+Q` | Quit |

#### TUI Screens

1. **Dashboard**: Overview of recent activity
2. **Execution**: Run new simulations
3. **Results**: View past simulation results
4. **Backends**: Manage simulators
5. **Settings**: Change configuration

### Running Simulations in TUI

1. Press `2` to go to Execution screen
2. Enter your circuit description (e.g., "bell state")
3. Choose backend (or leave as "auto")
4. Set number of shots
5. Press "Run" or hit `Enter`
6. Wait for results to appear

---

## Understanding Backends

### What is a Backend?

A **backend** is a quantum simulator - the actual software that runs your quantum circuit. Think of it as:
- Different calculators for different problems
- Some are faster, some are more accurate
- Proxima can use multiple backends

### Available Backends

| Backend Name | Description | Best For |
|-------------|-------------|----------|
| **lret** | Default, fast simulator | Small circuits (2-10 qubits) |
| **cirq** | Google's simulator | General purpose, flexible |
| **qiskit** | IBM's simulator | IBM ecosystem, widely used |
| **quest** | High-performance | Large circuits (15+ qubits) |
| **qsim** | CPU-optimized | Fast CPU simulations |
| **cuquantum** | GPU-accelerated | Very large circuits (needs NVIDIA GPU) |

### Listing Available Backends

```powershell
proxima backends list
# OR use shortcut
proxima be
```

### Checking Backend Status

```powershell
proxima backends status
```

This shows:
- ✓ Installed and available
- ○ Available but not installed
- ✗ Not available or error

### Selecting a Backend

#### Automatic Selection (Recommended for beginners)
```powershell
proxima run "bell state" --backend auto
```
Proxima picks the best one automatically.

#### Manual Selection
```powershell
proxima run "bell state" --backend cirq
```

### Setting Default Backend

```powershell
proxima config set backends.default_backend cirq
```

Now all simulations will use `cirq` by default unless you specify otherwise.

---

## Common Tasks

### Task 1: Run a Simple Simulation

```powershell
# Activate environment (if not already)
.\venv\Scripts\activate

# Run simulation
proxima run "bell state"
```

### Task 2: Compare Different Backends

```powershell
# Compare across all available backends
proxima compare "bell state" --all

# Compare specific backends
proxima compare "bell state" --backends lret cirq qiskit
```

### Task 3: View Simulation History

```powershell
# List last 20 simulations
proxima history list

# Show details of a specific result
proxima history show <result-id>
```

### Task 4: Change Settings

```powershell
# View current settings
proxima config show

# Change default backend
proxima config set backends.default_backend qiskit

# Change default number of shots
proxima config set backends.default_shots 2048

# Enable colored output
proxima config set general.color_enabled true
```

### Task 5: Export Results

```powershell
# Export to JSON
proxima export --format json --output results.json

# Export to CSV
proxima export --format csv --output results.csv
```

### Task 6: Use Interactive Shell

```powershell
# Start interactive mode
proxima interactive

# Inside the shell:
> run "bell state"
> backends list
> history list
> help run
> exit
```

### Task 7: Manage Sessions

Sessions help organize multiple simulations:

```powershell
# Create a new session
proxima session new "My Experiment"

# List all sessions
proxima session list

# Show session details
proxima session show <session-id>

# Resume a session
proxima session resume <session-id>
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "proxima: command not found"

**Problem**: Proxima is not installed or virtual environment is not activated.

**Solution**:
```powershell
cd C:\Proxima
.\venv\Scripts\activate
pip install -e .[all]
```

#### Issue 2: "No module named 'proxima'"

**Problem**: Virtual environment not activated.

**Solution**:
```powershell
cd C:\Proxima
.\venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

#### Issue 3: "Backend not available"

**Problem**: A backend is not installed.

**Solution 1**: Use automatic backend selection:
```powershell
proxima run "bell state" --backend auto
```

**Solution 2**: Install the missing backend:
```powershell
pip install cirq qiskit qiskit-aer
```

#### Issue 4: TUI won't start

**Problem**: Missing TUI dependencies.

**Solution**:
```powershell
pip install textual
# OR reinstall with UI support
pip install -e .[ui]
```

#### Issue 5: Slow performance

**Solutions**:
- Use a faster backend: `--backend lret` for small circuits
- Reduce shots: `--shots 512`
- Close other programs
- Use fewer qubits

#### Issue 6: Permission errors on Windows

**Solution**: Run PowerShell as Administrator:
1. Right-click PowerShell
2. Choose "Run as Administrator"
3. Navigate to Proxima directory and retry

#### Issue 7: "pip: command not found"

**Problem**: Python not installed or not in PATH.

**Solution**:
1. Reinstall Python from python.org
2. **Check** "Add Python to PATH" during installation
3. Restart PowerShell

### Getting More Help

#### 1. Run Diagnostics
```powershell
proxima doctor
```

#### 2. View Detailed Help
```powershell
proxima --help
proxima run --help
proxima backends --help
```

#### 3. Check Logs
```powershell
# View configuration location
proxima config show

# Logs are typically in:
# C:\Users\dell\.proxima\logs\
```

#### 4. Get Version Information
```powershell
proxima --version
python --version
```

---

## Quick Reference

### Essential Commands Cheat Sheet

```powershell
# ============================================
# SETUP & STATUS
# ============================================
.\venv\Scripts\activate              # Activate environment
proxima --version                    # Check version
proxima doctor                       # Run diagnostics
proxima status                       # Check status
proxima init                         # Initialize config

# ============================================
# RUNNING SIMULATIONS
# ============================================
proxima run "bell state"             # Basic run
proxima r "bell state"               # Shortcut
proxima bell                         # Quick Bell state
proxima ghz --qubits 3               # Quick GHZ state
proxima qft --qubits 4               # Quick QFT

# With options
proxima run "bell state" --backend cirq --shots 2000

# ============================================
# BACKENDS
# ============================================
proxima backends list                # List backends
proxima be                           # Shortcut
proxima backends status              # Check status

# ============================================
# CONFIGURATION
# ============================================
proxima config show                  # View settings
proxima cfg                          # Shortcut
proxima config set backends.default_backend cirq

# ============================================
# HISTORY & RESULTS
# ============================================
proxima history list                 # List history
proxima hist                         # Shortcut
proxima history show <id>            # View details
proxima export --format json         # Export results

# ============================================
# COMPARISON
# ============================================
proxima compare "bell state" --all   # Compare all backends
proxima cmp "bell state" --backends lret cirq

# ============================================
# USER INTERFACE
# ============================================
proxima ui                           # Launch TUI
python run_tui.py                    # Alternate TUI launch
proxima interactive                  # Interactive shell

# ============================================
# SESSIONS
# ============================================
proxima session list                 # List sessions
proxima sess                         # Shortcut
proxima session new "My Experiment"  # Create session
proxima session show <id>            # View session

# ============================================
# HELP
# ============================================
proxima --help                       # General help
proxima run --help                   # Command help
proxima aliases                      # List shortcuts
```

### Common Workflows

#### Workflow 1: Quick Test
```powershell
.\venv\Scripts\activate
proxima bell
```

#### Workflow 2: Compare Backends
```powershell
.\venv\Scripts\activate
proxima compare "bell state" --all
```

#### Workflow 3: Organized Session
```powershell
.\venv\Scripts\activate
proxima session new "Bell State Tests"
proxima run "bell state" --backend lret
proxima run "bell state" --backend cirq
proxima run "bell state" --backend qiskit
proxima session show <session-id>
```

#### Workflow 4: Visual Interface
```powershell
.\venv\Scripts\activate
python run_tui.py
# Use keyboard shortcuts to navigate
```

---

## Configuration Options

### Viewing Configuration

```powershell
proxima config show
```

### Important Settings

| Setting | What It Controls | Example |
|---------|-----------------|---------|
| `backends.default_backend` | Which simulator to use | `lret`, `cirq`, `qiskit` |
| `backends.default_shots` | Number of measurements | `1024`, `2048`, `4096` |
| `general.output_format` | Output style | `text`, `json`, `rich` |
| `general.color_enabled` | Colored output | `true`, `false` |
| `general.verbosity` | Detail level | `0` (quiet) to `3` (debug) |

### Changing Settings

```powershell
# Set default backend
proxima config set backends.default_backend cirq

# Set default shots
proxima config set backends.default_shots 2048

# Enable colors
proxima config set general.color_enabled true

# Set verbosity
proxima config set general.verbosity 1
```

### Resetting Configuration

```powershell
proxima config reset
```

---

## Next Steps

### After Mastering the Basics

1. **Learn More About Quantum Circuits**
   - Experiment with different circuits
   - Try different qubit counts
   - Understand quantum gates

2. **Explore Advanced Features**
   - Use agent files (automation)
   - Set up benchmarks
   - Use LLM integration for insights

3. **Read Additional Documentation**
   - Check `docs/` folder for advanced topics
   - Read `docs/user-guide/` for detailed guides
   - Explore `docs/api-reference/` for technical details

4. **Join the Community**
   - Report issues on GitHub
   - Contribute improvements
   - Share your experiences

---

## Tips for Success

### 💡 Best Practices

1. **Always activate the virtual environment** before using Proxima
2. **Start with small circuits** (2-4 qubits) to learn
3. **Use `auto` backend** until you understand the differences
4. **Run `proxima doctor`** if you encounter problems
5. **Use the TUI** if you're uncomfortable with command line
6. **Keep Proxima updated**: `pip install -e .[all] --upgrade`

### ⚡ Performance Tips

1. **For quick tests**: Use `lret` backend with fewer shots
2. **For accuracy**: Use more shots (2048+)
3. **For large circuits**: Use `quest` or `cuquantum` (if available)
4. **Close other programs** when running heavy simulations

### 🛡️ Safety Tips

1. **Always backup** important results before experimenting
2. **Use sessions** to organize your work
3. **Export results** regularly
4. **Don't run too many qubits** on slow computers (>15 qubits can be very slow)

---

## Glossary

**Backend**: A quantum simulator that executes quantum circuits.

**Circuit**: A quantum program made of quantum gates applied to qubits.

**Qubit**: A quantum bit, the basic unit of quantum information.

**Shot**: A single execution of a quantum circuit with measurement.

**Bell State**: A simple entangled quantum state between two qubits.

**GHZ State**: Generalization of Bell state to multiple qubits.

**QFT**: Quantum Fourier Transform, important quantum algorithm.

**TUI**: Terminal User Interface, a visual interface in the terminal.

**Virtual Environment**: Isolated Python environment for a project.

**Session**: A collection of related simulation runs.

---

## Getting Help

### Resources

- **Built-in Help**: `proxima --help`, `proxima [command] --help`
- **Documentation**: `docs/` folder in Proxima directory
- **Diagnostics**: `proxima doctor`
- **Status Check**: `proxima status`

### Contact

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Read `docs/` for advanced topics
- **Examples**: Check `docs/user-guide/benchmark-examples.md`

---

## Frequently Asked Questions

### Q: Do I need a quantum computer to use Proxima?
**A**: No! Proxima simulates quantum computers on your regular computer.

### Q: How many qubits can I simulate?
**A**: Depends on your computer. Most can handle 10-15 qubits comfortably. Each additional qubit roughly doubles memory requirements.

### Q: Is Proxima free?
**A**: Yes, Proxima is open source under the MIT license.

### Q: Can I use Proxima for research?
**A**: Yes! That's one of its main purposes.

### Q: What if I get errors?
**A**: Run `proxima doctor` first, then check the Troubleshooting section above.

### Q: Do I need to know quantum mechanics?
**A**: Basic understanding helps, but Proxima makes it easier to experiment and learn.

### Q: Can I write my own circuits?
**A**: Yes! You can describe circuits in natural language or use JSON format.

### Q: How do I update Proxima?
**A**: 
```powershell
.\venv\Scripts\activate
pip install -e .[all] --upgrade
```

---

## Summary

You've learned:
- ✅ What Proxima is and what it does
- ✅ How to install and set up Proxima
- ✅ How to run quantum simulations
- ✅ How to use the Terminal UI
- ✅ How to manage backends and settings
- ✅ How to troubleshoot common issues
- ✅ Quick reference commands

**Next**: Try running your first simulation!

```powershell
cd C:\Proxima
.\venv\Scripts\activate
proxima run "bell state"
```

**Happy Simulating! 🚀✨**

---

*This guide was created for Proxima v0.1.0. For the most up-to-date information, check the documentation in the `docs/` folder.*
