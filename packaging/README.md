# Proxima Agent Packaging Guide

# ==============================

This directory contains packaging and distribution templates for Proxima Agent.

## Distribution Channels

### 1. PyPI (Python Package Index)

The primary distribution channel. Users install with:

```bash
pip install proxima-agent
```

**Files involved:**

- `../pyproject.toml` - Package metadata and dependencies
- `../MANIFEST.in` - Source distribution includes
- `../src/proxima/py.typed` - Type hints marker (PEP 561)

**Release process:**

```bash
# Build
python -m build

# Upload to TestPyPI (testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### 2. Docker / Container Registry

Containerized distribution via GitHub Container Registry:

```bash
docker pull ghcr.io/proxima-project/proxima:latest
```

**Files involved:**

- `../Dockerfile` - Multi-stage build
- `../docker-compose.yml` - Development and production compose
- `../.dockerignore` - Build context exclusions

**Build process:**

```bash
# Build locally
docker build -t proxima-agent:latest .

# Build with compose
docker-compose build proxima
```

### 3. Homebrew (macOS/Linux)

**Files involved:**

- `homebrew/proxima.rb` - Formula template

**Setup a tap:**

1. Create repo: `github.com/proxima-project/homebrew-proxima`
2. Add formula to `Formula/proxima.rb`
3. Users install with: `brew tap proxima-project/proxima && brew install proxima`

**Update formula after release:**

1. Update version URL
2. Calculate SHA256: `shasum -a 256 proxima-agent-X.Y.Z.tar.gz`
3. Update formula hash

### 4. Standalone Binaries

Pre-built executables for users without Python.

**Built using:** PyInstaller (via GitHub Actions)

**Platforms:**

- Linux x86_64
- macOS x86_64 (Intel)
- macOS arm64 (Apple Silicon)
- Windows x86_64

**Manual build:**

```bash
pip install pyinstaller
pyinstaller --onefile --name proxima src/proxima/__main__.py
```

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)

- Runs on every push/PR
- Linting, type checking, testing
- Builds package and Docker image

### Release Workflow (`.github/workflows/release.yml`)

- Triggered by version tags (`v*.*.*`)
- Builds all distribution formats
- Publishes to PyPI and GHCR
- Creates GitHub release with binaries

## Version Management

Version is defined in `pyproject.toml`:

```toml
version = "0.1.0"
```

**Release steps:**

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit: `git commit -m "Release v0.1.0"`
4. Tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. Push: `git push origin main && git push origin v0.1.0`

Or use the build script:

```bash
python scripts/build.py release --version 0.1.0 --no-dry-run
```

## Checksums and Verification

All releases include SHA256 checksums in `checksums.sha256`.

Verify downloads:

```bash
sha256sum -c checksums.sha256
```

## Signing (Future)

Future releases may be signed with GPG or sigstore.
