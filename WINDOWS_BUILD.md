# Windows Build Compatibility Guide

> **Last Updated:** January 26, 2026  
> **Purpose:** Ensure Proxima works with any version of Visual Studio

---

## Overview

Proxima includes several quantum computing packages (like `qiskit-aer`) that contain C++ extensions requiring compilation on Windows. This guide explains how Proxima ensures compatibility with **any version of Visual Studio**, including:

- Visual Studio 2015 (v14.0)
- Visual Studio 2017 (v15.0)
- Visual Studio 2019 (v16.0)
- Visual Studio 2022 (v17.0)
- **Future versions** (auto-detected)

---

## Quick Start

### Recommended Installation

```powershell
# Navigate to Proxima directory
cd C:\path\to\Proxima

# Run the Windows installation script
.\scripts\install-windows.ps1
```

This script:
1. ✅ Automatically detects your Visual Studio version
2. ✅ Configures the build environment appropriately
3. ✅ Uses pre-built wheels when available
4. ✅ Falls back to source compilation only when necessary

### Alternative: Manual Installation

```powershell
# Step 1: Run build configuration
python scripts\configure_build.py

# Step 2: Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Step 3: Install Proxima (prefer binary wheels)
pip install -e .[all] --prefer-binary
```

---

## How It Works

### Automatic VS Detection

Proxima uses multiple methods to detect your Visual Studio installation:

1. **vswhere.exe** - Microsoft's official tool (most reliable)
2. **Environment Variables** - VSINSTALLDIR, VisualStudioVersion
3. **Common Paths** - Scans Program Files for VS installations

### Version Mapping

| Visual Studio | Internal Version | CMake Generator | MSVC Toolset |
|--------------|------------------|-----------------|--------------|
| VS 2015 | 14.0 | Visual Studio 14 2015 | v140 |
| VS 2017 | 15.0 | Visual Studio 15 2017 | v141 |
| VS 2019 | 16.0 | Visual Studio 16 2019 | v142 |
| VS 2022 | 17.0 | Visual Studio 17 2022 | v143 |
| VS 20XX (Future) | Auto-detected | Latest known | Latest known |

### Environment Variables Set

The configuration script sets these environment variables:

```
DISTUTILS_USE_SDK=1
MSSdk=1
CMAKE_GENERATOR=<detected generator>
PIP_PREFER_BINARY=1
```

---

## Troubleshooting

### Error: "Microsoft Visual C++ 14.0 or greater is required"

This error occurs when:
1. Visual Studio is not installed
2. The C++ workload is not installed
3. Environment is not configured correctly

**Solution:**

```powershell
# Option 1: Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Option 2: Use only pre-built packages
pip install -e .[all] --only-binary :all:
```

### Error: "CMake Error: Could not find CMAKE_C_COMPILER"

**Solution:**
1. Install Visual Studio with "Desktop development with C++"
2. Or install CMake separately: `winget install Kitware.CMake`

### Error: "error: Microsoft Visual C++ is required"

**Solution:**

```powershell
# Run the configuration script first
python scripts\configure_build.py

# Then install with binary preference
pip install -e .[all] --prefer-binary
```

### Specific Package Fails to Build

If a specific package (e.g., `qiskit-aer`) fails:

```powershell
# Install everything else first
pip install -e . --prefer-binary

# Then try the problematic package separately
pip install qiskit-aer --only-binary :all:

# If still failing, check for a compatible version
pip install qiskit-aer==0.14.2
```

---

## Future-Proofing

### How Future VS Versions Are Handled

When a newer version of Visual Studio is detected (e.g., VS 2024, 2026):

1. The detection script recognizes it as "newer than known"
2. It uses the VS 2022 configuration as a baseline
3. Environment variables are set for maximum compatibility
4. Binary wheels are preferred to avoid compilation

### Adding Support for New VS Versions

To add explicit support for a new VS version, edit `scripts/configure_build.py`:

```python
VS_VERSION_MAP = {
    "18": ("18.0", "Visual Studio 18 2024", "v144", 2024),  # Add this
    "17": ("17.0", "Visual Studio 17 2022", "v143", 2022),
    # ... existing versions
}
```

---

## Best Practices

### For End Users

1. **Always use the installation script** - It handles all configuration
2. **Prefer binary wheels** - They avoid compilation issues entirely
3. **Keep Visual Studio updated** - Newer versions have better compatibility

### For Developers

1. **Test on multiple VS versions** - Use GitHub Actions with different VS versions
2. **Provide pre-built wheels** - Upload to PyPI for common platforms
3. **Document minimum requirements** - But don't hardcode specific versions

---

## Related Files

| File | Purpose |
|------|---------|
| `scripts/configure_build.py` | Python build configuration script |
| `scripts/install-windows.ps1` | PowerShell installation script |
| `build_config.py` | VS detection and environment setup |
| `requirements-build.txt` | Build-time dependencies |
| `pyproject.toml` | Package configuration |

---

## Resources

- [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
- [Python Windows Compiler Guide](https://wiki.python.org/moin/WindowsCompilers)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [CMake Generator Reference](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html)
