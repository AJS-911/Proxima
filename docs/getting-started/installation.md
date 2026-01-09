# Installation

This guide covers all methods for installing Proxima on your system.

## System Requirements

| Requirement | Minimum               | Recommended |
| ----------- | --------------------- | ----------- |
| Python      | 3.10+                 | 3.12+       |
| RAM         | 4 GB                  | 16 GB       |
| Disk Space  | 500 MB                | 2 GB        |
| OS          | Linux, macOS, Windows | Any         |

## Quick Install (PyPI)

The simplest way to install Proxima:

```bash
pip install proxima-agent
```

To install with all optional dependencies:

```bash
pip install proxima-agent[all]
```

## Installation Options

### Option 1: From PyPI (Recommended)

```bash
# Core installation
pip install proxima-agent

# With LLM support (OpenAI, Anthropic)
pip install proxima-agent[llm]

# With TUI support (Textual terminal interface)
pip install proxima-agent[ui]

# With all features
pip install proxima-agent[all]

# For development
pip install proxima-agent[dev]
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# With development dependencies
pip install -e .[dev]
```

### Option 3: Using Docker

```bash
# Pull the official image
docker pull proxima-agent:latest

# Run interactively
docker run -it proxima-agent:latest proxima --help

# Run with volume mounts for persistence
docker run -it \
  -v ~/.proxima:/home/proxima/.proxima \
  -v $(pwd):/workspace \
  proxima-agent:latest proxima run "bell state"
```

### Option 4: Using Docker Compose

```bash
# Clone the repository
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima

# Start with docker-compose
docker-compose up -d

# Access the container
docker-compose exec proxima proxima --help
```

### Option 5: Homebrew (macOS/Linux)

```bash
# Add the tap (when available)
brew tap prthmmkhija1/proxima

# Install
brew install proxima
```

## Verify Installation

After installation, verify Proxima is working:

```bash
# Check version
proxima --version

# Show help
proxima --help

# List available backends
proxima backends list

# Run a quick test
proxima run --backend cirq "bell state test"
```

## Optional Dependencies

Proxima has several optional dependency groups:

| Group  | Description                       | Install Command                   |
| ------ | --------------------------------- | --------------------------------- |
| `llm`  | LLM providers (OpenAI, Anthropic) | `pip install proxima-agent[llm]`  |
| `ui`   | Terminal UI (Textual)             | `pip install proxima-agent[ui]`   |
| `dev`  | Development tools                 | `pip install proxima-agent[dev]`  |
| `docs` | Documentation tools               | `pip install proxima-agent[docs]` |
| `all`  | Everything                        | `pip install proxima-agent[all]`  |

## Quantum Backend Dependencies

Proxima supports multiple quantum simulation backends. Install them as needed:

```bash
# Cirq (Google)
pip install cirq

# Qiskit Aer (IBM)
pip install qiskit-aer

# LRET (Custom framework)
pip install git+https://github.com/kunal5556/LRET.git@feature/framework-integration
```

## Troubleshooting

### Common Issues

**Import Error: No module named 'proxima'**

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall
pip install -e .
```

**Permission Denied**

```bash
# Use --user flag
pip install --user proxima-agent

# Or use sudo (not recommended)
sudo pip install proxima-agent
```

**Textual TUI not working**

```bash
# Install with UI dependencies
pip install proxima-agent[ui]

# Check terminal compatibility
proxima ui check
```

**Backend not found**

```bash
# List available backends
proxima backends list

# Install missing backend
pip install cirq  # or qiskit-aer
```

### Getting Help

- **Documentation**: [https://proxima-agent.readthedocs.io](https://proxima-agent.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/prthmmkhija1/Pseudo-Proxima/issues)
- **Discussions**: [GitHub Discussions](https://github.com/prthmmkhija1/Pseudo-Proxima/discussions)
