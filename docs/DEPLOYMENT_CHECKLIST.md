# Proxima Deployment Checklist

> **Version:** 1.0  
> **Last Updated:** January 12, 2026  
> **Purpose:** Pre-release validation and deployment preparation for Proxima with new backends

---

## Overview

This checklist ensures all components are properly validated before releasing a new version of Proxima with QuEST, cuQuantum, and qsim backend support.

---

## Table of Contents

1. [Pre-Release Validation](#pre-release-validation)
2. [Release Artifacts](#release-artifacts)
3. [Post-Release Tasks](#post-release-tasks)

---

## Pre-Release Validation

### 1. Unit Tests

- [ ] All unit tests pass locally
  ```bash
  pytest tests/unit -v --tb=short
  ```

- [ ] Backend adapter unit tests pass
  ```bash
  pytest tests/backends/test_quest_adapter.py -v
  pytest tests/backends/test_cuquantum_adapter.py -v
  pytest tests/backends/test_qsim_adapter.py -v
  pytest tests/backends/test_backend_selection.py -v
  ```

- [ ] Test coverage meets threshold (≥80%)
  ```bash
  pytest --cov=src/proxima/backends --cov-report=html
  ```

### 2. Integration Tests

- [ ] All integration tests pass
  ```bash
  pytest tests/backends/test_integration.py -v
  pytest tests/integration -v
  ```

- [ ] Multi-backend comparison tests pass
  ```bash
  pytest tests/backends/test_integration.py::TestMultiBackendComparison -v
  ```

- [ ] Fallback logic tests pass
  ```bash
  pytest tests/backends/test_integration.py::TestFallbackLogicIntegration -v
  ```

### 3. Backend Auto-Selection

- [ ] Auto-selection works correctly
  ```bash
  proxima run --backend auto --verbose examples/bell_state.py
  ```

- [ ] Selection explains reasoning
  ```bash
  proxima run --backend auto --verbose examples/large_circuit.py
  ```

- [ ] Priority ordering is correct for each scenario:
  - [ ] State vector + GPU: cuQuantum preferred
  - [ ] State vector + CPU: qsim preferred  
  - [ ] Density matrix: QuEST preferred
  - [ ] Noisy circuits: QuEST/Qiskit preferred

### 4. Documentation

- [ ] All documentation files exist:
  - [ ] `docs/backends/quest-installation.md`
  - [ ] `docs/backends/quest-usage.md`
  - [ ] `docs/backends/backend-selection.md`
  - [ ] `docs/api-reference/backends/index.md`
  - [ ] `docs/user-guide/cli-reference.md`
  - [ ] `docs/migration/adding-backends-guide.md`

- [ ] Documentation is accurate and up-to-date
  - [ ] All code examples work
  - [ ] Configuration options are documented
  - [ ] CLI commands are correct

- [ ] API reference is complete
  - [ ] All public classes documented
  - [ ] All methods have docstrings
  - [ ] Type hints are present

### 5. Configuration Schema

- [ ] Configuration schema validated
  ```bash
  proxima config validate
  ```

- [ ] All new options have defaults
  ```bash
  proxima config show --defaults
  ```

- [ ] Invalid configs are rejected with clear messages
  ```bash
  proxima config set backends.quest.invalid_option value
  # Should show clear error
  ```

### 6. Performance Benchmarks

- [ ] Benchmark suite runs successfully
  ```bash
  pytest tests/backends/test_benchmark.py -v --benchmark-only
  ```

- [ ] Performance meets targets:
  - [ ] QuEST: Within 20% of native QuEST performance
  - [ ] cuQuantum: GPU utilization >70% for large circuits
  - [ ] qsim: Within 10% of native qsim performance

- [ ] No performance regressions vs previous version
  ```bash
  pytest tests/backends/test_benchmark.py --benchmark-compare
  ```

### 7. Error Messages

- [ ] Error messages are user-friendly
- [ ] Installation instructions included in dependency errors
- [ ] Resource errors suggest alternatives
- [ ] Unsupported operation errors explain limitations

### 8. Platform Testing

- [ ] Tests pass on Ubuntu/Linux
  ```bash
  # Run on Ubuntu
  pytest tests/ -v
  ```

- [ ] Tests pass on Windows
  ```bash
  # Run on Windows
  pytest tests/ -v
  ```

- [ ] Tests pass on macOS
  ```bash
  # Run on macOS
  pytest tests/ -v
  ```

### 9. GPU Testing (NVIDIA Hardware Required)

- [ ] cuQuantum tests pass on GPU hardware
  ```bash
  pytest tests/backends/test_cuquantum*.py -v
  # Requires NVIDIA GPU
  ```

- [ ] GPU memory management works correctly
  - [ ] Memory is freed after execution
  - [ ] Out-of-memory errors are handled gracefully

- [ ] QuEST GPU mode works (if built with CUDA)
  ```bash
  proxima run --backend quest --gpu test_circuit.py
  ```

### 10. Validation Against Known Results

- [ ] Bell state validation passes
  ```bash
  pytest tests/backends/test_validation.py::TestBellStateValidation -v
  ```

- [ ] GHZ state validation passes
  ```bash
  pytest tests/backends/test_validation.py::TestGHZStateValidation -v
  ```

- [ ] Cross-backend fidelity >0.9999
  ```bash
  pytest tests/backends/test_validation.py::TestCrossBackendValidation -v
  ```

---

## Release Artifacts

### 1. Python Package

- [ ] Version number updated in `pyproject.toml`
  ```toml
  [project]
  version = "X.Y.Z"
  ```

- [ ] CHANGELOG updated with new features
  ```markdown
  ## [X.Y.Z] - YYYY-MM-DD
  ### Added
  - QuEST backend support
  - cuQuantum backend support
  - qsim backend support
  - Intelligent backend auto-selection
  ```

- [ ] Build succeeds
  ```bash
  python -m build
  ```

- [ ] Package installs correctly
  ```bash
  pip install dist/proxima-X.Y.Z-py3-none-any.whl
  ```

- [ ] Upload to PyPI (test first)
  ```bash
  # Test PyPI
  twine upload --repository testpypi dist/*
  
  # Production PyPI
  twine upload dist/*
  ```

### 2. Docker Images

- [ ] CPU-only Docker image builds
  ```bash
  docker build -t proxima:X.Y.Z -f Dockerfile.cpu .
  ```

- [ ] CPU Docker image works
  ```bash
  docker run proxima:X.Y.Z proxima backends list
  ```

- [ ] GPU Docker image builds (if applicable)
  ```bash
  docker build -t proxima:X.Y.Z-gpu -f Dockerfile.gpu .
  ```

- [ ] GPU Docker image works with nvidia-docker
  ```bash
  docker run --gpus all proxima:X.Y.Z-gpu proxima backends list
  ```

- [ ] Push to container registry
  ```bash
  docker push proxima:X.Y.Z
  docker push proxima:X.Y.Z-gpu
  ```

### 3. Binary Releases (if applicable)

- [ ] Windows executable builds
- [ ] macOS binary builds
- [ ] Linux binary builds
- [ ] Binaries are signed (if required)

### 4. Documentation Website

- [ ] Documentation builds without errors
  ```bash
  mkdocs build --strict
  ```

- [ ] Documentation deployed to hosting
  ```bash
  mkdocs gh-deploy
  ```

- [ ] All links work (no 404s)
- [ ] Search functionality works
- [ ] Mobile-friendly layout verified

### 5. Release Notes

- [ ] Release notes written with:
  - [ ] New features summary
  - [ ] Migration instructions
  - [ ] Breaking changes (if any)
  - [ ] Known issues
  - [ ] Contributors credited

- [ ] GitHub release created with:
  - [ ] Tag: vX.Y.Z
  - [ ] Release notes attached
  - [ ] Binary artifacts attached (if applicable)

---

## Post-Release Tasks

### 1. Monitoring

- [ ] Monitor GitHub issues for new reports
  - Set up issue templates for backend-specific problems
  - Tag issues with appropriate backend labels

- [ ] Monitor PyPI download statistics
  ```bash
  # Check PyPI stats
  pip-download-stats proxima
  ```

- [ ] Check CI/CD pipeline is green

### 2. User Feedback

- [ ] Update backend compatibility matrix based on user feedback
  - Track which backends work on which platforms
  - Document any platform-specific issues

- [ ] Collect performance benchmarks from users
  - Create a benchmarks discussion thread
  - Document real-world performance data

- [ ] Address common questions in FAQ

### 3. Documentation Updates

- [ ] Iterate on documentation based on user questions
  - Add clarifications where needed
  - Update examples if issues found

- [ ] Add any missing documentation discovered post-release

### 4. Follow-up Releases

- [ ] Plan patch releases for critical bugs
- [ ] Collect feature requests for next version
- [ ] Update roadmap based on feedback

---

## Quick Validation Commands

Run these commands for a quick pre-release check:

```bash
# 1. Run all tests
pytest tests/ -v --tb=short

# 2. Check code quality
black --check src tests
ruff check src tests
mypy src --ignore-missing-imports

# 3. Test backends
proxima backends list --all
proxima backends test --all

# 4. Run quick validation
proxima run --backend auto examples/bell_state.py
proxima compare --backends quest,qsim examples/bell_state.py

# 5. Build package
python -m build

# 6. Build docs
mkdocs build --strict
```

---

## Rollback Plan

If critical issues are discovered post-release:

### Immediate Actions

1. **Assess severity**
   - Is it a security issue?
   - Does it break existing functionality?
   - How many users are affected?

2. **Communicate**
   - Post GitHub issue with known problem
   - Update release notes with warning
   - Notify users via appropriate channels

3. **Rollback if necessary**
   ```bash
   # Yank problematic version from PyPI
   pip index versions proxima
   # Contact PyPI support for yank if needed
   
   # Release patched version quickly
   # Increment patch version: X.Y.Z+1
   ```

### Post-Mortem

1. Document what went wrong
2. Add missing tests to prevent recurrence
3. Update this checklist if needed

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| Reviewer | | | |
| QA | | | |
| Release Manager | | | |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-12 | Initial checklist for new backends release |

---

## Notes

- This checklist should be completed for every release
- All items must be checked before proceeding to release
- If an item cannot be completed, document the reason and get approval to skip
- Keep this document updated as processes change
