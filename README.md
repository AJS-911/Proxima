<p align="center">
  <img src="https://img.shields.io/badge/Quantum-Simulation-blueviolet?style=for-the-badge" alt="Quantum Simulation"/>
  <img src="https://img.shields.io/badge/AI-Powered-orange?style=for-the-badge" alt="AI Powered"/>
  <img src="https://img.shields.io/badge/Multi--Backend-Support-green?style=for-the-badge" alt="Multi-Backend"/>
</p>

<h1 align="center">🌌 Proxima</h1>

<p align="center">
  <strong>Intelligent Quantum Simulation Orchestration Framework</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://pypi.org/project/proxima-agent/"><img src="https://img.shields.io/pypi/v/proxima-agent.svg" alt="PyPI version"/></a>
  <a href="https://github.com/prthmmkhija1/Pseudo-Proxima/actions"><img src="https://img.shields.io/github/actions/workflow/status/prthmmkhija1/Pseudo-Proxima/ci.yml?label=CI" alt="CI"/></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🎯 What is Proxima?

**Proxima** is an intelligent quantum simulation orchestration framework that simplifies quantum computing experimentation. Write what you want in plain English, and Proxima handles the rest—selecting the optimal backend, managing resources, and providing human-readable insights.

```bash
# It's this simple!
proxima run "create a 3-qubit entangled state"
```

---

## ✨ Features

| Feature                      | Description                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| 🔀 **Multi-Backend Support** | Seamlessly switch between Cirq, Qiskit Aer, and LRET simulators   |
| 🧠 **Intelligent Selection** | Auto-selects the best backend based on your circuit requirements  |
| ⏯️ **Execution Control**     | Pause, resume, abort, and rollback simulations on demand          |
| 📊 **Resource Monitoring**   | Real-time CPU, memory tracking with fail-safe mechanisms          |
| 🤖 **LLM Integration**       | Connect to OpenAI, Anthropic, or local Ollama models for insights |
| 📈 **Result Interpretation** | Get human-readable explanations of quantum results                |
| ⚖️ **Backend Comparison**    | Run identical circuits across backends and compare performance    |
| 🎨 **Beautiful TUI**         | Interactive terminal interface for visual exploration             |

---

## 🚀 Quick Start

### Install

```bash
pip install proxima-agent[all]
```

### Initialize

```bash
proxima init
```

### Run Your First Simulation

```bash
proxima run --backend cirq "bell state with 2 qubits"
```

**That's it!** You'll see real-time progress, resource usage, and results with AI-powered insights.

---

## 📦 Installation

### PyPI (Recommended)

```bash
# Base installation
pip install proxima-agent

# Full installation with all extras
pip install proxima-agent[all]

# Specific extras
pip install proxima-agent[llm]    # LLM integrations
pip install proxima-agent[ui]     # Terminal UI
pip install proxima-agent[dev]    # Development tools
```

### Docker

```bash
docker pull ghcr.io/proxima-project/proxima:latest
docker run --rm -it ghcr.io/proxima-project/proxima:latest run "bell state"
```

### From Source

```bash
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima
pip install -e ".[all]"
```

---

## 💻 Usage

### Basic Commands

```bash
# List available backends
proxima backends list

# Run a simulation
proxima run --backend cirq "quantum teleportation"

# Let AI choose the best backend
proxima run --backend auto "5-qubit GHZ state"

# Compare backends
proxima compare --backends cirq,qiskit "bell state"
```

### Interactive TUI

Launch the beautiful terminal interface:

```bash
proxima ui
```

<details>
<summary>📸 TUI Preview</summary>

```
┌─────────────────────────────────────────────────────────┐
│  🌌 PROXIMA - Quantum Simulation Dashboard              │
├─────────────────────────────────────────────────────────┤
│  [1] Dashboard    [2] Execute    [3] Results           │
│  [4] Backends     [5] Config     [?] Help    [q] Quit  │
├─────────────────────────────────────────────────────────┤
│  Status: Ready                                          │
│  Memory: ████████░░ 78%                                 │
│  Active Sessions: 0                                     │
└─────────────────────────────────────────────────────────┘
```

</details>

### Agent Files (Batch Processing)

Create a file `experiment.md`:

```markdown
# My Experiment

## Task 1: Bell State

- backend: cirq
- shots: 1024
- circuit: bell state

## Task 2: Compare Results

- compare: cirq, qiskit
- circuit: bell state
```

Run it:

```bash
proxima agent run experiment.md
```

---

## 🏗️ Architecture

```
proxima/
├── cli/              # Command-line interface
├── core/             # Domain logic & orchestration
├── backends/         # Quantum backend adapters
│   ├── cirq/         # Google Cirq integration
│   ├── qiskit/       # IBM Qiskit integration
│   └── lret/         # LRET simulator
├── intelligence/     # AI/ML components
│   ├── llm_router/   # LLM provider abstraction
│   └── insights/     # Result interpretation
├── tui/              # Terminal user interface
└── resources/        # Resource monitoring
```

---

## 📚 Documentation

| Resource               | Link                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| 📖 Full Documentation  | [docs/](./docs/)                                                                             |
| 🚀 Getting Started     | [docs/getting-started/](./docs/getting-started/)                                             |
| 🔧 Configuration Guide | [docs/user-guide/configuration.md](./docs/user-guide/configuration.md)                       |
| 🧩 Backend Development | [docs/developer-guide/backend-development.md](./docs/developer-guide/backend-development.md) |
| 📋 API Reference       | [docs/api-reference/](./docs/api-reference/)                                                 |

---

## 🔧 Supported Backends

| Backend        | Type                         | Max Qubits | Features                    |
| -------------- | ---------------------------- | ---------- | --------------------------- |
| **Cirq**       | State Vector, Density Matrix | 30+        | Fast, Google ecosystem      |
| **Qiskit Aer** | State Vector, Density Matrix | 30+        | IBM ecosystem, noise models |
| **LRET**       | Custom                       | Varies     | Lightweight, extensible     |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
git clone https://github.com/prthmmkhija1/Pseudo-Proxima.git
cd Pseudo-Proxima
pip install -e ".[all]"

# Run tests
pytest

# Run linting
ruff check src/ tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Architectural inspiration from [OpenCode AI](https://github.com/opencode-ai/opencode) and [Crush](https://github.com/charmbracelet/crush)
- The quantum computing community for invaluable resources

---

<p align="center">
  <sub>Built with ❤️ for the quantum computing community</sub>
</p>

<p align="center">
  <a href="https://github.com/prthmmkhija1/Pseudo-Proxima/stargazers">⭐ Star us on GitHub</a>
</p>
