# Proxima: Intelligent Quantum Simulation Orchestration Framework

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/proxima-agent.svg)](https://pypi.org/project/proxima-agent/)
[![Docker Image](https://img.shields.io/docker/v/proxima-project/proxima-agent?label=docker)](https://ghcr.io/proxima-project/proxima)
[![CI](https://github.com/proxima-project/proxima/actions/workflows/ci.yml/badge.svg)](https://github.com/proxima-project/proxima/actions/workflows/ci.yml)

Proxima is an intelligent quantum simulation orchestration framework that provides a unified interface for running quantum simulations across multiple backends with advanced features like automatic backend selection, resource monitoring, and intelligent result interpretation.

## Features

- **Multi-Backend Support**: LRET, Cirq (DensityMatrix + StateVector), Qiskit Aer (DensityMatrix + StateVector)
- **Intelligent Backend Selection**: Automatic selection with explanations
- **Execution Control**: Start, Abort, Pause, Resume, Rollback
- **Resource Awareness**: Memory and CPU monitoring with fail-safe mechanisms
- **Explicit Consent**: User confirmation for critical operations
- **LLM Integration**: Support for local and remote AI models
- **Result Interpretation**: Human-readable insights and analytics
- **Multi-Backend Comparison**: Run identical simulations across backends
- **Execution Transparency**: Real-time progress and timing display

## Installation

### From PyPI (Recommended)

```bash
# Install the base package
pip install proxima-agent

# Install with all optional dependencies (LLM, TUI, dev tools)
pip install proxima-agent[all]

# Install specific extras
pip install proxima-agent[llm]    # LLM integrations (OpenAI, Anthropic)
pip install proxima-agent[ui]     # Terminal UI (Textual)
pip install proxima-agent[dev]    # Development tools
```

### Using Docker

```bash
# Pull the latest image
docker pull ghcr.io/proxima-project/proxima:latest

# Run with Docker
docker run --rm -it ghcr.io/proxima-project/proxima:latest --help

# Run a simulation
docker run --rm -it \
  -v ~/.proxima:/home/proxima/.proxima \
  ghcr.io/proxima-project/proxima:latest \
  run --backend cirq "bell state"

# Using Docker Compose
docker-compose up -d proxima
docker-compose run proxima backends list
```

### Using Homebrew (macOS/Linux)

```bash
# Add the tap
brew tap proxima-project/proxima

# Install
brew install proxima

# Verify installation
proxima version
```

### From Source

```bash
# Clone the repository
git clone https://github.com/proxima-project/proxima.git
cd proxima

# Install in development mode
pip install -e ".[all]"

# Or use the build script
python scripts/build.py build
```

### Standalone Binaries

Download pre-built binaries from the [Releases](https://github.com/proxima-project/proxima/releases) page:

- **Linux**: `proxima-linux-x86_64`
- **macOS Intel**: `proxima-darwin-x86_64`
- **macOS Apple Silicon**: `proxima-darwin-arm64`
- **Windows**: `proxima-windows-x86_64.exe`

```bash
# Make executable (Linux/macOS)
chmod +x proxima-linux-x86_64

# Run
./proxima-linux-x86_64 --help
```

## Quick Start

```bash
# Initialize configuration
proxima init

# Show version
proxima version

# List available backends
proxima backends list

# Run a simulation
proxima run --backend cirq "bell state"

# Compare across backends
proxima compare --backends cirq,qiskit "quantum teleportation"
```

## Docker Quick Start

```bash
# Run with Docker Compose
docker-compose up -d

# Execute commands
docker-compose run proxima backends list
docker-compose run proxima run --backend cirq "entanglement"

# With local LLM support (Ollama)
docker-compose --profile llm up -d
docker-compose run proxima run --llm ollama "analyze circuit"

# Development mode
docker-compose --profile dev up -d
docker-compose exec proxima-dev pytest
```

## Project Structure

```
proxima/
├── src/proxima/          # Main package
│   ├── cli/              # Command-line interface
│   ├── core/             # Core domain logic
│   ├── backends/         # Backend adapters
│   ├── intelligence/     # AI/ML components
│   ├── resources/        # Resource management
│   ├── data/             # Data handling
│   ├── config/           # Configuration
│   └── utils/            # Utilities
├── tests/                # Test suites
├── configs/              # Configuration files
├── scripts/              # Build and release scripts
├── packaging/            # Distribution packaging
└── docs/                 # Documentation
```

## Configuration

Proxima supports multiple configuration sources (in priority order):

1. Command-line arguments
2. Environment variables (PROXIMA\_\*)
3. User config file (~/.proxima/config.yaml)
4. Project config file (./proxima.yaml)
5. Default values

## Development

```bash
# Using the build script (recommended)
python scripts/build.py all          # Run all checks
python scripts/build.py test         # Run tests
python scripts/build.py lint         # Run linting
python scripts/build.py build        # Build package
python scripts/build.py release --version 0.1.0  # Prepare release

# Or using PowerShell on Windows
.\scripts\build.ps1 all
.\scripts\build.ps1 test -Coverage

# Manual commands
pytest                               # Run tests
pytest --cov=proxima                 # With coverage
black src/ tests/                    # Format code
ruff check src/ tests/               # Lint code
mypy src/                            # Type check
mkdocs serve                         # Serve docs locally
```

## Building & Releasing

```bash
# Build Python package
python -m build

# Build Docker image
docker build -t proxima-agent:latest .

# Prepare a release (dry run)
python scripts/build.py release --version 0.2.0

# Execute release
python scripts/build.py release --version 0.2.0 --no-dry-run
```

## Architecture

Proxima follows a layered modular architecture:

1. **Presentation Layer**: CLI, TUI (future), Web API (future)
2. **Orchestration Layer**: Planner, Executor, State Manager
3. **Intelligence Layer**: LLM Router, Backend Selector, Insight Engine
4. **Resources & Safety Layer**: Memory Monitor, Consent Manager, Execution Control
5. **Backend Abstraction Layer**: Unified adapter interface
6. **Data & Output Layer**: Result storage, comparison, export

## Roadmap

- **Phase 1** (Weeks 1-4): Foundation & Core Infrastructure ✅
- **Phase 2** (Weeks 5-9): Backend Integration ✅
- **Phase 3** (Weeks 10-14): Intelligence Features ✅
- **Phase 4** (Weeks 15-18): Safety & Resource Management ✅
- **Phase 5** (Weeks 19-23): Advanced Features ✅
- **Phase 6** (Weeks 24-27): Production Ready ✅

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our [contributing guidelines](docs/developer-guide/contributing.md) first.

See the [CHANGELOG](CHANGELOG.md) for version history.

## Credits

Architectural inspiration from:

- [OpenCode AI](https://github.com/opencode-ai/opencode)
- [Crush (Charmbracelet)](https://github.com/charmbracelet/crush)

Proxima is an independent implementation, not a fork or derivative.

---

## 📚 Complete Usage Guide - Step by Step

### Step 1: Installation & Setup

**Option A: Using pip (Easiest)**

```bash
# Install Proxima with all features
pip install proxima-agent[all]

# Initialize configuration
proxima init

# Verify installation
proxima version
proxima backends list
```

**Option B: Clone from GitHub**

```bash
# Clone the repository
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima

# Install in development mode
pip install -e ".[all]"

# Verify
proxima version
```

---

### Step 2: Check Available Backends

```bash
# List all quantum backends
proxima backends list

# Check details of a specific backend
proxima backends info cirq
proxima backends info qiskit
```

**What you'll see:**

- ✅ Available backends (Cirq, Qiskit, LRET)
- Backend capabilities (max qubits, noise support, GPU support)
- Version information

---

### Step 3: Run Your First Quantum Simulation

**Basic Circuit Execution:**

```bash
# Run on Cirq backend (automatic)
proxima run --backend cirq "2-qubit bell state"

# Run on Qiskit backend
proxima run --backend qiskit "3-qubit GHZ state"

# Let Proxima choose the best backend automatically
proxima run --backend auto "quantum circuit"
```

**What happens:**

1. ⏱️ Timer starts showing elapsed time
2. 🔍 Backend validates your circuit
3. 💾 Checks memory/CPU resources
4. ✅ Asks for your consent
5. 🚀 Executes the simulation
6. 📊 Shows results with insights

---

### Step 4: Compare Multiple Backends

```bash
# Compare Cirq vs Qiskit performance
proxima compare --backends cirq,qiskit "bell state"

# Compare all available backends
proxima compare --backends cirq,qiskit,lret "entanglement circuit"

# Compare with custom shots
proxima compare --backends cirq,qiskit --shots 2048 "teleportation"
```

**What you get:**

- ⚡ Execution time for each backend
- 💾 Memory usage comparison
- 📈 Result agreement percentage
- 🏆 Recommended backend with explanation

---

### Step 5: Use the Terminal UI (TUI)

```bash
# Launch interactive TUI
proxima ui launch

# Or just
proxima ui
```

**TUI Features:**

- **Screen 1 (Dashboard)**: Overview of all backends and recent runs
- **Screen 2 (Execution)**: Real-time execution progress
- **Screen 3 (Configuration)**: Edit settings interactively
- **Screen 4 (Results)**: View past results with visualizations
- **Screen 5 (Backends)**: Compare backend capabilities

**Keyboard shortcuts:**

- `1-5`: Switch between screens
- `q`: Quit
- `?`: Help menu

---

### Step 6: Use Agent Files (proxima_agent.md)

Create a file called `my_experiment.md`:

```markdown
# Quantum Experiment

## Task 1: Circuit Execution

- backend: cirq
- shots: 1024
- circuit: bell state preparation

## Task 2: Backend Comparison

- compare: cirq, qiskit
- circuit: same as Task 1
- shots: 2048

## Task 3: Export Results

- format: json
- output: results.json
```

Then run:

```bash
proxima agent run my_experiment.md
```

**What happens:**

- Proxima reads all tasks from the file
- Executes them sequentially
- Asks consent before each task
- Generates combined report at the end

---

### Step 7: Control Execution Flow

**Pause and Resume:**

```bash
# Start a long-running simulation
proxima run --backend qiskit "complex 10-qubit circuit" &

# Pause it (in another terminal or using Ctrl+Z)
proxima session pause <session-id>

# Resume later
proxima session resume <session-id>

# Abort if needed
proxima session abort <session-id>
```

**Rollback Feature:**

```bash
# If something goes wrong, rollback to last checkpoint
proxima session rollback <session-id>
```

---

### Step 8: Work with LLM (AI Integration)

**Local LLM (Ollama):**

```bash
# Run with local LLM for insights
proxima run --llm ollama --backend cirq "bell state"
```

**Remote LLM (OpenAI/Anthropic):**

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Run with AI-powered insights
proxima run --llm openai --backend cirq "quantum circuit"

# Proxima will ask consent before using remote LLM
```

**What LLM does:**

- Explains circuit behavior in plain English
- Suggests optimizations
- Interprets measurement results
- Recommends best practices

---

### Step 9: Export and Analyze Results

**Export in Multiple Formats:**

```bash
# Export as JSON
proxima run --backend cirq "bell state" --export json --output results.json

# Export as CSV
proxima run --backend cirq "bell state" --export csv --output data.csv

# Export as HTML report
proxima run --backend cirq "bell state" --export html --output report.html

# Export as Excel (with multiple sheets)
proxima run --backend cirq "bell state" --export xlsx --output analysis.xlsx
```

**View Past Results:**

```bash
# List execution history
proxima history list

# Show details of specific run
proxima history show <run-id>

# Export history to file
proxima history export --format json --output history.json
```

---

### Step 10: Configuration Management

**View Configuration:**

```bash
# Show current configuration
proxima config show

# Show specific setting
proxima config get backends.default_backend
```

**Update Configuration:**

```bash
# Set default backend
proxima config set backends.default_backend cirq

# Set memory threshold
proxima config set resources.memory_threshold 0.85

# Enable dry-run mode by default
proxima config set general.dry_run true
```

**Configuration File Location:**

- User config: `~/.proxima/config.yaml`
- Project config: `./proxima.yaml`

---

### Step 11: Advanced Features

**Dry Run Mode (Plan without executing):**

```bash
# See what would happen without running
proxima run --dry-run --backend cirq "bell state"
```

**Force Mode (Skip consent prompts):**

```bash
# Auto-approve all operations (use carefully!)
proxima run --force --backend qiskit "circuit"
```

**Verbose Output:**

```bash
# Show detailed logs
proxima -vvv run --backend cirq "bell state"

# Or quiet mode
proxima --quiet run --backend cirq "bell state"
```

**Custom Output Format:**

```bash
# JSON output (for scripting)
proxima --output json backends list

# Rich formatted output
proxima --output rich run --backend cirq "bell state"
```

---

### Step 12: Monitor Resources

**Real-time Monitoring:**

```bash
# Show resource usage while running
proxima run --backend qiskit "large circuit" --monitor
```

**Resource Thresholds:**
Proxima automatically warns you at:

- 60% memory usage: ⚠️ Warning
- 80% memory usage: ⚠️ Strong warning
- 95% memory usage: 🛑 Critical - execution may be blocked

---

### Step 13: Session Management

**List Active Sessions:**

```bash
proxima session list
```

**Session Details:**

```bash
proxima session status <session-id>
```

**Session Persistence:**
All sessions are automatically saved to `~/.proxima/sessions/` and can be resumed after restart.

---

### Step 14: Plugin System (Advanced)

**Create Custom Plugins:**

```python
# ~/.proxima/plugins/my_plugin.py
from proxima.plugins.base import BasePlugin

class MyPlugin(BasePlugin):
    def on_execution_start(self, context):
        print("Execution starting!")

    def on_execution_complete(self, context, result):
        print(f"Done! Result: {result}")
```

**Enable Plugin:**

```yaml
# ~/.proxima/config.yaml
plugins:
  enabled:
    - my_plugin
  search_paths:
    - ~/.proxima/plugins
```

---

### Step 15: Troubleshooting

**Common Issues:**

1. **Backend not available:**

```bash
# Install missing backend
pip install cirq  # For Cirq
pip install qiskit qiskit-aer  # For Qiskit
```

2. **Memory errors:**

```bash
# Lower memory threshold
proxima config set resources.memory_threshold 0.7
```

3. **Permission errors:**

```bash
# Check config directory permissions
chmod 755 ~/.proxima
```

4. **View logs:**

```bash
# Check log file
cat ~/.proxima/logs/proxima.log

# Or with verbose mode
proxima -vvv run --backend cirq "circuit"
```

---

### 📖 Full Command Reference

```bash
# Core Commands
proxima init                          # Initialize configuration
proxima version                       # Show version
proxima run [OPTIONS] DESCRIPTION     # Run simulation
proxima compare [OPTIONS]             # Compare backends

# Backend Management
proxima backends list                 # List all backends
proxima backends info NAME            # Backend details
proxima backends test NAME            # Test backend

# Configuration
proxima config show                   # View config
proxima config get KEY                # Get setting
proxima config set KEY VALUE          # Set setting
proxima config reset                  # Reset to defaults

# Agent Files
proxima agent run FILE                # Execute agent.md file
proxima agent validate FILE           # Validate agent.md
proxima agent preview FILE            # Preview execution plan

# History
proxima history list                  # Show past runs
proxima history show ID               # Run details
proxima history export [OPTIONS]      # Export history
proxima history clear                 # Clear history

# Sessions
proxima session list                  # Active sessions
proxima session status ID             # Session details
proxima session pause ID              # Pause session
proxima session resume ID             # Resume session
proxima session abort ID              # Abort session
proxima session rollback ID           # Rollback session

# UI
proxima ui launch                     # Launch TUI
proxima ui check                      # Check TUI dependencies

# Global Options
--config PATH                         # Config file path
--backend NAME                        # Select backend
--output FORMAT                       # Output format (text/json/rich)
--verbose, -v                         # Verbose output (stackable: -vvv)
--quiet, -q                           # Quiet mode
--dry-run                             # Plan only
--force, -f                           # Skip confirmations
```

---

### 🎯 Example Workflows

**Workflow 1: Quick Experiment**

```bash
proxima run --backend auto "3-qubit GHZ state"
```

**Workflow 2: Detailed Comparison**

```bash
proxima compare --backends cirq,qiskit --shots 4096 "bell state" --export html --output comparison.html
```

**Workflow 3: Batch Processing with Agent File**

```bash
# Create experiment.md with multiple tasks
proxima agent run experiment.md --export json --output results.json
```

**Workflow 4: Production Pipeline**

```bash
# 1. Dry run to validate
proxima run --dry-run --backend cirq "circuit"

# 2. Execute with monitoring
proxima run --backend cirq "circuit" --monitor --export xlsx --output prod_results.xlsx

# 3. Review in TUI
proxima ui
```

---

### 🆘 Getting Help

- **Documentation**: Full docs at `docs/` or run `mkdocs serve`
- **Command help**: `proxima --help` or `proxima COMMAND --help`
- **GitHub Issues**: https://github.com/prthmmkhija1/Pseudo-Proxima/issues
- **Examples**: See `examples/` directory in the repo

---

**🎉 You're now ready to use Proxima for quantum simulations!**
