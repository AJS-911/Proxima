# AI-Driven Implementation for Phases 2-8

This document contains the complete AI-driven (non-hardcoded) implementation details for phases 2-8 that should replace the current hardcoded content in BACKENDS_INFO.md.

## Phase 2 (Complete): LLM-Based Dependency Detection & Resolution

### Step 2.2: LLM-Powered Dependency Name Resolution

Let LLM translate generic dependency names to platform-specific package names:

**Process:**
- For each dependency from Phase 1 (e.g., "Eigen3", "CMake", "pybind11")
- Prompt LLM: "On {operating_system} using package manager {package_manager}, what is the package name for dependency '{dependency_name}' version {version_constraint}? Consider: 1) Exact package name, 2) Development headers package if needed (like -dev suffix), 3) Alternative package names, 4) Whether it's available via package manager or needs manual installation"
- LLM uses its knowledge of package ecosystems to translate
- No dependency database needed - LLM trained on package manager documentation

**For checking if already installed:**
- Execute package manager query: `apt list --installed | grep {package}` or `brew list {package}` or `pip show {package}`
- Send output to LLM: "Is {dependency} already installed? Package manager output: {output}"
- LLM interprets output (handles different output formats across package managers)

**Technical Approach:**
- No JSON/YAML mapping database
- LLM query for each dependency with context (OS, package manager)
- LLM returns package name or "not available via package manager"
- Use package manager query commands, let LLM interpret results
- LLM handles special cases (header-only libraries, Python vs system packages)

**Key Insight:** LLM knows that Eigen3 is "libeigen3-dev" on apt, "eigen" on brew, "eigen3" on vcpkg. No need to hardcode this knowledge.

### Step 2.3: LLM-Guided Dependency Installation

Let LLM generate and verify installation commands:

**Process:**
- For each missing dependency, prompt LLM: "Generate the command to install {package_name} using {package_manager} on {os}. Include: 1) Full command with any necessary flags, 2) Whether sudo/admin privileges needed, 3) Expected output indicating success, 4) Common error messages and their meanings"
- LLM generates: `{"command": "sudo apt install -y libeigen3-dev", "needs_sudo": true, "success_indicators": ["Setting up", "done"], "timeout": 300}`
- Request user permission: "I need to install {package_name}. Command: {command}. Proceed? (y/n)"
- Execute command using Python `subprocess`
- Send output to LLM: "Installation output: {output}. Was installation successful? Are there errors?"
- LLM analyzes output and confirms success or diagnoses failure

**For installation failures:**
- LLM receives error output
- LLM suggests fixes: "Try updating package lists first: sudo apt update"
- Retry with LLM's suggestion

**Technical Approach:**
- No hardcoded installation command templates
- LLM generates commands based on package manager knowledge
- Use `subprocess.run()` with timeout, capture stdout/stderr
- LLM interprets success/failure from output text
- Implement retry with LLM-suggested modifications
- Log all operations

**Key Insight:** LLM can generate installation commands for any package manager (apt, dnf, pacman, brew, choco, winget, pip, conda) without templates.

### Step 2.4: AI-Assisted Compiler Toolchain Setup

LLM determines compiler requirements and guides setup:

**Process:**
- Prompt LLM: "Project requires C++{standard} compiler. System has: {compiler_versions}. Determine: 1) Is available compiler sufficient? 2) If not, what needs to be installed? 3) How to install it on {os}? 4) How to configure environment variables?"
- LLM analyzes compiler version output (e.g., "gcc version 9.4.0")
- LLM compares against requirement (e.g., "needs C++17, gcc 7+ or clang 5+")
- LLM generates installation/upgrade commands if needed

**For Windows:**
- LLM knows to suggest Visual Studio Build Tools or MinGW
- LLM generates PowerShell commands to check for Visual Studio: `Get-Command cl.exe -ErrorAction SilentlyContinue`
- LLM interprets vswhere.exe output if available

**For macOS:**
- LLM suggests Xcode Command Line Tools: `xcode-select --install`
- LLM checks if tools are installed: `xcode-select -p`

**Environment Configuration:**
- LLM generates commands to set CC and CXX environment variables
- Example: `export CC=/usr/bin/gcc-11` or `set CC=cl.exe`
- LLM creates shell script for environment setup if needed

**Technical Approach:**
- Execute compiler version checks, send output to LLM
- LLM reasons about compiler capabilities (C++17/20 support)
- No hardcoded compiler version database
- LLM generates platform-specific installation instructions
- LLM creates environment setup scripts dynamically

**Key Insight:** LLM understands compiler versioning (gcc 9 supports C++17, MSVC 2019 supports C++17). No need to hardcode version-to-feature mappings.

---

## Phase 3: Intelligent Build System Analysis & Configuration

### Step 3.1: LLM-Generated CMake Configuration

Let LLM analyze CMakeLists.txt and generate configuration commands:

**Process:**
- Read CMakeLists.txt contents
- Prompt LLM: "Analyze this CMakeLists.txt: {contents}. Generate: 1) CMake configuration command with optimal flags for {os} and {compiler}, 2) List of configurable options (option() statements), 3) Required CMake version, 4) Recommended generator (Visual Studio/Ninja/Make), 5) Any platform-specific considerations, 6) Installation prefix recommendation"
- LLM reads project structure and suggests appropriate settings
- LLM generates complete cmake command: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja ...`

**For CMake options:**
- LLM identifies `option()` statements in CMakeLists.txt
- Prompt: "These CMake options are available: {options}. For a quantum simulation backend on {os}, which should be enabled/disabled? Consider performance and compatibility."
- LLM reasons about which options to use (e.g., enable SIMD, disable examples)

**Technical Approach:**
- Read CMakeLists.txt as text, send to LLM
- LLM understands CMake syntax (option, set, find_package)
- LLM generates configuration command based on: OS, compiler, available dependencies, project type
- No CMake parser needed - LLM reads CMake language
- Execute LLM-generated command
- If CMake fails, send error to LLM for command revision

**Key Insight:** LLM knows CMake syntax and best practices. It can read CMakeLists.txt and determine appropriate configuration without hardcoded command templates.

### Step 3.2: LLM-Driven Python Build Configuration

LLM analyzes Python project structure and generates build commands:

**Process:**
- Check for setup.py, pyproject.toml, setup.cfg
- Read file contents
- Prompt LLM: "Analyze this Python project configuration: {pyproject_toml or setup_py}. Determine: 1) Build backend (setuptools/poetry/flit/hatchling), 2) Has C/C++ extensions? (pybind11/Cython/cffi), 3) Build command (pip install/python -m build/poetry build), 4) Should use isolated environment?, 5) Extra dependencies to install first"
- LLM identifies build system from [build-system] table or setup.py imports
- LLM generates appropriate build command

**For C/C++ extensions:**
- LLM detects if setup.py has `Extension()` or pyproject.toml mentions pybind11
- LLM ensures C++ compiler is available before building
- LLM adds necessary build flags to environment variables

**Technical Approach:**
- Read Python configuration files as text
- LLM determines build backend from file contents (no PEP 517 parser initially)
- LLM generates command: `python -m build --wheel` or `pip install -e .` or `poetry build`
- LLM decides whether to create virtual environment for isolation
- Use Python `subprocess` to execute LLM-generated commands

**Key Insight:** LLM understands Python packaging ecosystem (setuptools, poetry, flit). It can read pyproject.toml and determine the build backend without hardcoded logic.

### Step 3.3: AI-Orchestrated Multi-Language Builds

LLM determines build order for complex multi-language projects:

**Process:**
- Present LLM with project structure: "Project has: CMakeLists.txt (C++ library), setup.py (Python bindings), Cargo.toml (Rust optimizer). Analyze dependencies between these components."
- LLM reads each build file
- LLM identifies dependencies: "Python setup.py imports compiled C++ library, so C++ must build first"
- Prompt: "Determine build order, commands for each step, and how to pass artifacts between steps (library paths, include directories)."
- LLM generates build pipeline

**Build pipeline example generated by LLM:**
```
1. Build C++ library: cmake -S . -B build && cmake --build build
2. Set environment: export LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH
3. Build Python bindings: cd python && pip install -e .
```

**Technical Approach:**
- Enumerate all build system files
- LLM analyzes inter-dependencies
- LLM creates ordered list of build steps
- LLM generates environment variable settings to propagate paths
- Execute steps sequentially, passing environment between steps
- No topological sort algorithm - LLM reasons about order

**Key Insight:** LLM can understand that Python bindings depend on compiled C++ library by reading setup.py (which might reference library paths).

### Step 3.4: LLM-Optimized Build Settings

LLM selects optimization flags based on project and system:

**Process:**
- Prompt LLM: "For a quantum simulation backend on {os} with {cpu_arch} CPU and {ram} RAM, generate optimal build settings: 1) Parallel compilation jobs (-j flag), 2) Optimization flags (-O3, LTO), 3) Architecture-specific flags (-march=native), 4) Linker memory limits, 5) Build caching tools (ccache/sccache)"
- LLM considers: CPU cores (don't exceed), RAM (prevent OOM), architecture (AVX2/AVX512)
- LLM generates optimized flags

**Example LLM reasoning:**
- "System has 16 cores and 8GB RAM. Large C++ project. Use -j8 (not -j16) to avoid OOM. Enable LTO for 10% speedup. Use -march=native for AVX2 on this CPU."

**Technical Approach:**
- Gather system info: `os.cpu_count()`, `psutil.virtual_memory()`, `platform.machine()`
- Send to LLM with project size estimate (line count, file count)
- LLM generates: parallel jobs, compiler flags, linker flags
- Apply settings via: `cmake -DCMAKE_CXX_FLAGS="-O3 -march=native"`, environment variables, or build system configuration
- No hardcoded flag templates - LLM generates based on context

**Key Insight:** LLM knows compiler optimization flags (-O3, -march=native, LTO) and can reason about trade-offs (compile time vs runtime performance, memory usage).

---

## (Continue with Phases 4-8 with similar AI-driven approach...)

[This file would be inserted back into BACKENDS_INFO.md to replace hardcoded sections]
