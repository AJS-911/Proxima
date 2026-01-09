# Contributing to Proxima Agent

Thank you for your interest in contributing to Proxima Agent! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

### Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/proxima.git
   cd proxima
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\Activate   # Windows
   ```

3. **Install in development mode**

   ```bash
   pip install -e ".[all]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

### Development Workflow

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Run quality checks:

   ```bash
   python scripts/build.py lint
   python scripts/build.py typecheck
   python scripts/build.py test --coverage
   ```

4. Commit with a descriptive message:

   ```bash
   git commit -m "feat: add quantum teleportation support"
   ```

5. Push and create a Pull Request

## Coding Standards

### Python Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type Checker**: MyPy
- **Python Version**: 3.11+

Run formatting:

```bash
black src/ tests/
ruff check --fix src/ tests/
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Formatting changes
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

Examples:

```
feat: add Qiskit Aer density matrix simulator
fix: correct probability normalization in Cirq adapter
docs: update installation instructions for Docker
```

### Type Hints

All functions should have type hints:

```python
def calculate_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculate fidelity between two quantum states."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def execute_simulation(
    circuit: QuantumCircuit,
    backend: str,
    shots: int = 1000,
) -> ExecutionResult:
    """Execute a quantum circuit on the specified backend.

    Args:
        circuit: The quantum circuit to execute.
        backend: Name of the backend to use.
        shots: Number of measurement shots.

    Returns:
        ExecutionResult containing measurements and metadata.

    Raises:
        BackendError: If the backend is unavailable.
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=proxima --cov-report=html

# Specific test file
pytest tests/unit/test_backends.py

# Specific test
pytest tests/unit/test_backends.py::test_cirq_state_vector
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place end-to-end tests in `tests/e2e/`
- Use fixtures from `tests/conftest.py`

```python
import pytest
from proxima.backends.cirq_adapter import CirqAdapter

class TestCirqAdapter:
    def test_state_vector_simulation(self, sample_circuit):
        adapter = CirqAdapter()
        result = adapter.execute(sample_circuit, simulator_type="state_vector")

        assert result.backend == "cirq"
        assert result.simulator_type == "state_vector"
        assert result.execution_time_ms > 0
```

## Adding New Features

### Adding a New Backend

1. Create adapter in `src/proxima/backends/`:

   ```python
   # src/proxima/backends/new_adapter.py
   from .base import BackendAdapter, ExecutionResult

   class NewAdapter(BackendAdapter):
       name = "new-backend"

       def execute(self, circuit, options) -> ExecutionResult:
           ...
   ```

2. Register in `src/proxima/backends/registry.py`

3. Add tests in `tests/unit/test_backends.py`

4. Update documentation in `docs/developer-guide/adding-backends.md`

### Adding a CLI Command

1. Create command in `src/proxima/cli/commands/`:

   ```python
   # src/proxima/cli/commands/new_command.py
   import typer

   app = typer.Typer()

   @app.command()
   def new_action():
       """Description of the command."""
       ...
   ```

2. Register in `src/proxima/cli/main.py`

3. Add tests and documentation

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted and linted
- [ ] Type hints are complete
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for features/fixes)

### PR Description

Include:

- What changes were made
- Why the changes were needed
- How to test the changes
- Screenshots (for UI changes)

### Review Process

1. Automated CI checks must pass
2. At least one maintainer review required
3. Address review comments
4. Squash and merge when approved

## Documentation

### Building Docs Locally

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

Visit http://localhost:8000

### Documentation Structure

- `docs/getting-started/` - Installation, quickstart
- `docs/user-guide/` - Usage guides
- `docs/developer-guide/` - Contributing, architecture
- `docs/api-reference/` - Auto-generated API docs

## Questions?

- Open a [GitHub Discussion](https://github.com/proxima-project/proxima/discussions)
- Check existing [Issues](https://github.com/proxima-project/proxima/issues)
- Read the [Documentation](https://proxima.readthedocs.io)

Thank you for contributing! 🎉
