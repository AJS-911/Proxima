# QuEST Backend Installation Guide

> **Version:** 1.0  
> **Last Updated:** January 12, 2026  
> **Backend:** QuEST (Quantum Exact Simulation Toolkit)

---

## Overview

QuEST is a high-performance quantum circuit simulator written in C++ that supports both state vector and density matrix simulations. It offers optional GPU acceleration via CUDA and distributed computing via MPI.

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 18.04, macOS 10.15 | Windows 11, Ubuntu 22.04, macOS 13+ |
| **RAM** | 8 GB | 32 GB+ (for >20 qubits) |
| **CPU** | x86_64 with SSE4 | Multi-core with AVX2/AVX512 |
| **GPU** | - | NVIDIA GPU with CUDA 11.2+ (optional) |
| **Python** | 3.9+ | 3.11+ |

### Build Dependencies (for source installation)

- C++ compiler with C++11 support (GCC 5.0+, Clang 3.4+, or MSVC 2015+)
- CMake 3.16 or higher
- Eigen3 library (for linear algebra)
- OpenMP (optional, for parallel execution)
- CUDA Toolkit 11.2+ (optional, for GPU support)
- MPI library (optional, for distributed computing)

---

## Installation Methods

### Method 1: Install Pre-built Python Bindings (Recommended)

The easiest way to get started is using the pre-built Python bindings:

```bash
# Install pyquest-cffi from PyPI
pip install pyquest-cffi>=0.9.0

# Verify installation
python -c "import pyquest; print('QuEST installed successfully!')"
```

### Method 2: Install via Conda

```bash
# Create conda environment (recommended)
conda create -n proxima-quest python=3.11
conda activate proxima-quest

# Install pyquest
pip install pyquest-cffi
```

### Method 3: Build from Source (for GPU Support)

If you need GPU acceleration, build QuEST from source:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/QuEST-Kit/QuEST.git
cd QuEST
```

#### Step 2: Configure CMake

**For CPU-only build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**For GPU build (CUDA):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DGPUACCELERATED=ON \
         -DGPU_COMPUTE_CAPABILITY=75  # Adjust for your GPU
```

**For distributed build (MPI):**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DDISTRIBUTED=ON
```

#### Step 3: Build and Install

```bash
# Build
cmake --build . --config Release -j$(nproc)

# Install (may require sudo)
cmake --install .
```

#### Step 4: Install Python Bindings

```bash
# Navigate to Python bindings directory
cd ../QuEST/pyquest-cffi

# Install
pip install .
```

---

## Windows-Specific Instructions

### Using Visual Studio

1. Install Visual Studio 2019 or 2022 with C++ workload
2. Install CMake (add to PATH)
3. Open PowerShell and run:

```powershell
# Clone repository
git clone https://github.com/QuEST-Kit/QuEST.git
cd QuEST

# Create build directory
mkdir build; cd build

# Configure (CPU only)
cmake .. -G "Visual Studio 17 2022" -A x64

# Build
cmake --build . --config Release
```

### Using MSVC with CUDA

1. Install CUDA Toolkit from NVIDIA
2. Ensure CUDA bin directory is in PATH
3. Configure with CUDA enabled:

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 `
         -DGPUACCELERATED=ON `
         -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
```

---

## Verification

After installation, verify QuEST works correctly:

```python
import pyquest

# Create QuEST environment
env = pyquest.createQuESTEnv()

# Create a quantum register with 3 qubits
qureg = pyquest.createQureg(3, env)

# Apply Hadamard gate to qubit 0
pyquest.hadamard(qureg, 0)

# Apply CNOT gate (control=0, target=1)
pyquest.controlledNot(qureg, 0, 1)

# Measure probabilities
prob_0 = pyquest.calcProbOfOutcome(qureg, 0, 0)
prob_1 = pyquest.calcProbOfOutcome(qureg, 0, 1)

print(f"Probability of qubit 0 being |0⟩: {prob_0:.4f}")
print(f"Probability of qubit 0 being |1⟩: {prob_1:.4f}")

# Cleanup
pyquest.destroyQureg(qureg, env)
pyquest.destroyQuESTEnv(env)

print("QuEST verification complete!")
```

---

## Proxima Integration Verification

Once QuEST is installed, verify it works with Proxima:

```bash
# Check if Proxima detects QuEST
proxima backends list

# Get detailed QuEST info
proxima backends info quest

# Run a simple test
proxima run --backend quest examples/bell_state.py
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: `ModuleNotFoundError: No module named 'pyquest'`

**Cause:** pyquest-cffi not installed or wrong Python environment

**Solution:**
```bash
# Check Python environment
which python  # or `where python` on Windows

# Reinstall
pip install --upgrade pyquest-cffi
```

#### 2. CUDA Error: `cudaErrorNoDevice`

**Cause:** No NVIDIA GPU found or CUDA drivers not installed

**Solution:**
- Verify GPU is present: `nvidia-smi`
- Install/update NVIDIA drivers
- For CPU fallback, QuEST will automatically use CPU mode

#### 3. Memory Error: `std::bad_alloc`

**Cause:** Insufficient RAM for the number of qubits

**Solution:**
- Reduce qubit count
- Use density matrix mode only for small circuits
- Memory requirement: 2^n × 16 bytes for state vector (n = qubits)

#### 4. Build Error: `Eigen not found`

**Cause:** Eigen3 library not installed

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Windows (vcpkg)
vcpkg install eigen3:x64-windows
```

---

## Performance Tuning

### OpenMP Thread Configuration

```python
import os

# Set number of threads before importing pyquest
os.environ['OMP_NUM_THREADS'] = '8'  # Adjust based on CPU cores

import pyquest
```

### Memory Estimation

Calculate memory requirements before running:

| Qubits | State Vector | Density Matrix |
|--------|-------------|----------------|
| 10 | 16 KB | 16 MB |
| 15 | 512 KB | 16 GB |
| 20 | 16 MB | 16 TB |
| 25 | 512 MB | - |
| 30 | 16 GB | - |

---

## Next Steps

- [QuEST Usage Guide](quest-usage.md) - Learn how to use QuEST with Proxima
- [Backend Selection Guide](backend-selection.md) - Understand when to use QuEST
- [API Reference](../api-reference/backends/quest.md) - Detailed API documentation

---

## Resources

- [QuEST GitHub Repository](https://github.com/QuEST-Kit/QuEST)
- [QuEST Documentation](https://quest-kit.github.io/QuEST/)
- [pyquest-cffi PyPI](https://pypi.org/project/pyquest-cffi/)
- [QuEST Tutorial Paper](https://arxiv.org/abs/1802.08032)
