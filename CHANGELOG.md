# Changelog

All notable changes to Proxima Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Preparing for initial release

---

## [0.1.0] - 2026-01-09

### Added

#### Core Features

- **CLI Framework**: Full command-line interface built with Typer
  - `proxima init` - Initialize configuration
  - `proxima version` - Display version information
  - `proxima config` - Configuration management commands
  - `proxima backends list` - List available quantum backends
  - `proxima run` - Execute simulations (foundation)
  - `proxima compare` - Multi-backend comparison (foundation)

#### Backend Integration

- **LRET Backend**: Integration with LRET quantum simulation framework
- **Cirq Backend**: Full support for Google's Cirq library
  - Density Matrix simulator
  - State Vector simulator
- **Qiskit Aer Backend**: Full support for IBM's Qiskit Aer
  - Density Matrix simulator
  - State Vector simulator
- **Backend Registry**: Plugin-based backend discovery and management
- **Result Normalization**: Unified result format across all backends

#### Configuration System

- Hierarchical configuration (CLI > ENV > User Config > Defaults)
- Pydantic-based settings validation
- YAML configuration file support
- Environment variable support (PROXIMA\_\* prefix)

#### Logging & Observability

- Structured logging with structlog
- Multiple output formats (text, JSON, rich)
- Configurable log levels
- Execution timing and transparency

#### State Management

- Finite State Machine for execution control
- States: IDLE, PLANNING, READY, RUNNING, PAUSED, COMPLETED, ABORTED, ERROR
- State persistence for session recovery

#### Resource Management

- Memory monitoring with psutil
- Resource estimation before execution
- Configurable memory thresholds
- Execution timeout support

#### Safety Features

- Consent management for sensitive operations
- Force execute option with warnings
- Explicit user confirmation for LLM usage

#### Intelligence Layer (Foundation)

- LLM Router architecture (local and remote support)
- Backend auto-selection with explanation
- Insight engine for result interpretation
- Support for OpenAI, Anthropic, Ollama, LM Studio

#### Data & Export

- Result storage system
- Export to CSV format
- Export to XLSX format with openpyxl
- Multi-backend comparison aggregation

### Infrastructure

- Full test suite with pytest
- GitHub Actions CI/CD pipeline
- Docker support with multi-stage builds
- PyPI packaging ready
- MkDocs documentation

### Documentation

- User guide
- Developer guide
- API reference (auto-generated)
- Getting started tutorial
- Configuration reference

---

## Version History

### Versioning Scheme

Proxima follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: New functionality (backwards-compatible)
- **PATCH** version: Bug fixes (backwards-compatible)

### Pre-release Labels

- `alpha` - Early development, unstable API
- `beta` - Feature complete, testing phase
- `rc` - Release candidate, final testing

### Release Cadence

- **Patch releases**: As needed for critical fixes
- **Minor releases**: Monthly during active development
- **Major releases**: When significant breaking changes are required

---

## Migration Guides

### Migrating from 0.x to 1.0 (Future)

Migration guides will be provided for each major version upgrade.

---

## Links

- [Documentation](https://proxima.readthedocs.io)
- [GitHub Repository](https://github.com/proxima-project/proxima)
- [Issue Tracker](https://github.com/proxima-project/proxima/issues)
- [PyPI Package](https://pypi.org/project/proxima-agent/)

[Unreleased]: https://github.com/proxima-project/proxima/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/proxima-project/proxima/releases/tag/v0.1.0
