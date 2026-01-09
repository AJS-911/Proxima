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
