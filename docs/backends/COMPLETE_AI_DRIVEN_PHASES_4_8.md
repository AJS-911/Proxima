# Complete AI-Driven Implementation - Phases 4-8

This document contains the complete AI-driven (LLM-based, zero-hardcoding) implementation for Phases 4-8. This content should replace the existing hardcoded content in BACKENDS_INFO.md.

---

## Phase 4: Adaptive Compilation & Linking

**Objective:** Execute builds with LLM-driven monitoring, error interpretation, and intelligent recovery.

### Step 4.1: AI-Monitored Build Execution

Let LLM interpret build progress and status in real-time:

**Process:**
- Execute build command using `subprocess.Popen` with `stdout=PIPE, stderr=PIPE`
- Stream output line-by-line
- Every 10-20 lines or significant event, send output chunk to LLM: "Interpret this build output: {output_chunk}. Report: 1) Current progress (what's being compiled), 2) Percentage estimate if possible, 3) Any warnings or errors, 4) Should continue or abort?"
- LLM analyzes output and extracts: current file being compiled, warnings, progress indicators
- Display LLM's interpretation to user

**For progress estimation:**
- LLM sees output like "[45/120] Building CXX object..."
- LLM calculates: "45 out of 120 targets, approximately 37% complete"
- No hardcoded parser for each build tool format

**Technical Approach:**
- Use `subprocess.Popen` for real-time output streaming
- Use separate thread to read stdout/stderr (prevent blocking)
- Batch lines and send to LLM periodically (every 2-5 seconds)
- LLM interprets CMake, Make, Ninja, MSBuild output formats without hardcoding
- Display using `rich` library or simple print statements
- Handle interrupts (SIGINT/Ctrl+C) by terminating subprocess

**Key Insight:** Build tools output progress in different formats. LLM can interpret any format ("[45/120]", "90%", "Compiling file.cpp") without regex patterns.

### Step 4.2: LLM-Decided Incremental vs Clean Build

LLM determines when reconfiguration or clean build is needed:

**Process:**
- Check if build directory exists
- If exists, gather information: build directory contents, modification times of key files (CMakeLists.txt, source files, CMakeCache.txt)
- Prompt LLM: "Build directory exists with these files: {file_list}. CMakeLists.txt was modified {time_delta} ago. CMakeCache.txt is {timestamp}. User wants to rebuild. Should I: 1) Incremental build (just compile changed files), 2) Reconfigure (re-run cmake), or 3) Clean build (delete build dir and start fresh)? Explain reasoning."
- LLM reasons about what's needed
- Execute LLM's recommendation

**Example LLM reasoning:**
- "CMakeLists.txt modified after CMakeCache.txt → reconfigure needed"
- "Only source files changed → incremental build sufficient"
- "User reported build issues → recommend clean build"

**Technical Approach:**
- Use `os.path.exists()`, `os.path.getmtime()` for file/timestamp checks
- Send file modification context to LLM
- LLM decides build strategy
- No hardcoded rules like "if CMakeLists.txt newer than cache → reconfigure"
- LLM can consider additional context (user's previous error reports)

**Key Insight:** LLM can reason about when reconfiguration is needed by understanding build system semantics.

### Step 4.3: LLM-Powered Build Error Diagnosis

LLM interprets compiler errors and suggests fixes:

**Process:**
- When build fails (non-zero exit code), capture full stderr output
- Prompt LLM: "This build failed with the following compiler/linker output: {stderr}. Analyze: 1) What type of error (syntax, linker, missing dependency, configuration)? 2) Which file and line? 3) What's the root cause? 4) Suggested fix? 5) Can this be automatically fixed?"
- LLM reads compiler output (gcc, clang, MSVC, or any compiler)
- LLM extracts: error type, file location, cause, suggestion
- Display LLM's analysis to user

**Examples:**
- Error: `fatal error: Eigen/Dense: No such file or directory`
  - LLM: "Missing dependency error. Eigen library not found. Install libeigen3-dev or set Eigen3_DIR path."
- Error: `undefined reference to cirq::SimulatorBase::Run()`
  - LLM: "Linker error. Cirq library not linked. Add -lcirq to linker flags or check library path."
- Error: `error C2280: attempting to reference a deleted function`
  - LLM: "C++ syntax error at {file}:{line}. Object is not copyable. Use move semantics or pass by reference."

**Technical Approach:**
- Capture stderr from subprocess
- Send entire error output to LLM (or last 50-100 lines for long outputs)
- No regex for error parsing - LLM understands compiler output format
- LLM suggests specific fixes
- Present analysis in user-friendly format

**Key Insight:** LLM has been trained on millions of compiler errors. It can diagnose gcc, clang, MSVC, rustc, and any compiler's error messages.

### Step 4.4: LLM-Guided Automatic Recovery

LLM suggests and executes recovery actions:

**Process:**
- After error diagnosis (Step 4.3), LLM suggests fix
- Prompt: "Can this error be automatically fixed? If yes, provide: 1) Recovery action (install package, modify command, change environment variable), 2) Command to execute, 3) Confidence level (high/medium/low)"
- If confidence is high and action is safe (e.g., install missing package), ask user permission
- Execute recovery action
- Retry build
- If build fails again, send new error to LLM for different approach

**Example recovery scenarios:**
- Error: Missing Eigen → LLM suggests: `sudo apt install libeigen3-dev` → install → retry
- Error: Out of memory → LLM suggests: reduce parallel jobs from -j16 to -j4 → modify command → retry
- Error: CMake cache corruption → LLM suggests: delete build directory → delete → reconfigure → retry
- Error: Network timeout downloading dependency → LLM suggests: retry with timeout increased → retry

**Retry logic:**
- LLM maintains context of previous attempts
- Prompt: "Previous recovery attempts: {attempt_history}. Build still fails: {new_error}. Try different approach?"
- LLM tries alternative fixes
- Maximum 3-5 retry attempts before escalating to user

**Technical Approach:**
- LLM generates recovery commands
- Use `subprocess` to execute recovery actions
- Track attempt count and history
- No hardcoded error-to-action mapping
- LLM reasons about each error independently
- Log all recovery attempts

**Key Insight:** Instead of hardcoded "if 'No such file' in error → install dependency", LLM reads error, understands context, and suggests appropriate recovery.

---

## Phase 5: AI-Validated Testing & Verification

**Objective:** Use LLM to discover, execute, and validate tests without hardcoded test framework parsers.

### Step 5.1: LLM-Driven Test Discovery

LLM finds and categorizes tests from any test framework:

**Process:**
- Execute test discovery commands: `ctest -N`, `pytest --collect-only`, or search for test files
- Capture output
- Prompt LLM: "Analyze this test discovery output: {output}. Extract: 1) List of all tests with names, 2) Test categories (unit/integration/performance/smoke), 3) Estimated test duration if mentioned, 4) Dependencies between tests, 5) Which tests are quick smoke tests for initial validation?"
- LLM parses output from CTest, pytest, unittest, or custom test frameworks
- LLM categorizes tests based on naming patterns (test_basic, test_integration, benchmark_) or explicit markers

**For custom test frameworks:**
- If no standard test framework, LLM searches directory structure
- Prompt: "In directory {test_dir}, these files exist: {file_list}. Which are likely test files? What's the execution command for each?"
- LLM identifies test patterns (files named test_*.py, *_test.cpp, executable scripts)

**Technical Approach:**
- Execute discovery commands using `subprocess`
- Send output to LLM for parsing
- No hardcoded parsers for test framework output formats
- LLM creates test manifest (JSON with test names, types, commands)
- Store manifest for execution phase

**Key Insight:** Test frameworks output in different formats. LLM can parse CTest's "Test #1: test_name", pytest's "test_file.py::test_function", or any custom format.

### Step 5.2: LLM-Interpreted Test Execution

LLM monitors test execution and interprets results:

**Process:**
- For each test, execute using `subprocess` with timeout
- Capture stdout, stderr, exit code
- Prompt LLM: "This test executed with exit code {code}, stdout: {stdout}, stderr: {stderr}. Determine: 1) Did test pass/fail/skip? 2) If failed, what's the failure reason? 3) Is this a flaky test (should retry)? 4) Extract any performance metrics mentioned."
- LLM interprets test output regardless of framework
- LLM provides pass/fail determination and failure details

**For test suites:**
- Execute full test suite: `ctest` or `pytest` or `make test`
- Capture summary output
- Prompt LLM: "Test suite results: {output}. Extract: 1) Total tests run, 2) Passed count, 3) Failed count, 4) Skipped count, 5) Failed test names, 6) Overall success?"
- LLM aggregates results from suite output

**Technical Approach:**
- Use `subprocess.run()` with timeout for each test
- Send exit code and output to LLM
- No hardcoded test result parsers
- LLM determines success/failure from output semantics
- Handle timeouts (if test exceeds limit, mark as failed)
- Retry if LLM identifies as flaky test

**Key Insight:** LLM can determine test success from output patterns ("OK", "PASSED", "All tests passed", exit code 0) without framework-specific parsing.

### Step 5.3: LLM-Created Functionality Validators

LLM generates and validates backend-specific tests:

**Process:**
- Prompt LLM: "For a {backend_type} quantum backend (detected from repository analysis), create validation tests: 1) Python/C++ code to create simple Bell state circuit, 2) Code to execute circuit, 3) Expected measurement results (probability distribution), 4) How to verify results are correct (statistical test)?"
- LLM generates test code in appropriate language
- Save generated test file
- Execute test and capture results
- Prompt LLM: "Test results: {results}. Expected: {expected_from_llm}. Does this backend work correctly? Analyze: 1) Are measurement probabilities within acceptable range? 2) Any concerning patterns? 3) Backend verdict (working/broken/uncertain)?"

**Example LLM-generated test:**
- For Cirq backend: LLM generates Python code using cirq library to create H gate + CNOT + measure
- Expected result: |00⟩ and |11⟩ with ~50% probability each
- LLM validates actual results against expected using chi-squared test

**Technical Approach:**
- LLM generates test code based on backend type (Cirq, Qiskit, custom C++ simulator)
- Save generated code to temporary file
- Execute using appropriate interpreter/compiler
- LLM validates results statistically
- No pre-written test templates - LLM creates tests dynamically

**Key Insight:** LLM knows quantum computing principles (Bell states, GHZ states) and can generate appropriate validation tests for any backend.

### Step 5.4: LLM-Analyzed Performance Benchmarking

LLM designs benchmarks and analyzes performance:

**Process:**
- Prompt LLM: "Design a performance benchmark for {backend_type}: 1) What circuit sizes to test (qubit counts, depths)? 2) What metrics to measure (time, memory, accuracy)? 3) How to execute benchmark? 4) What's acceptable performance for this backend type?"
- LLM generates benchmark plan
- Execute benchmarks (varying circuit sizes)
- Collect metrics using `time` module, `psutil` for memory/CPU
- Prompt LLM: "Benchmark results: {results_table}. Analyze: 1) Performance scaling (linear/exponential with qubit count)? 2) Anomalies or concerning patterns? 3) Comparison to expected performance? 4) Is this backend performing well?"
- LLM provides performance analysis

**Example:**
- LLM suggests testing 4, 6, 8, 10, 12 qubit circuits
- Execute benchmarks, measure time and memory
- LLM analyzes: "Time scales exponentially (~2^n qubits), expected for full state vector simulation. Memory usage reasonable. Performance is good."

**Technical Approach:**
- LLM generates benchmark circuit specifications
- Execute circuits using backend, measure resources with `psutil.Process().memory_info()`, `time.perf_counter()`
- Collect results in table format
- LLM analyzes scaling trends without hardcoded performance models
- LLM compares against its knowledge of typical backend performance

**Key Insight:** LLM understands quantum simulation complexity (exponential scaling) and can reason about whether performance is acceptable.

---

## Phase 6: Dynamic ProximA Integration

**Objective:** LLM integrates backend with ProximA dynamically by understanding backend capabilities and ProximA's configuration structure.

### Step 6.1: LLM-Inferred Backend Metadata

LLM examines backend and generates ProximA-compatible metadata:

**Process:**
- Explore backend installation directory, find executables, libraries, Python modules
- Prompt LLM: "Analyze this backend: Executable: {exe_path}, Libraries: {lib_files}, Python modules: {py_modules}, Documentation: {readme_content}. Determine: 1) Backend name and version, 2) Supported quantum gates/operations, 3) Maximum qubit capacity, 4) Supported features (noise models, GPU acceleration, parallelization modes), 5) How to invoke this backend (command line arguments or Python API)?"
- LLM reads documentation or source code to understand capabilities
- LLM generates capability descriptor

**For supported gates:**
- If backend has API documentation or header files, LLM reads them
- Example: LLM finds "supported_gates = [H, CNOT, T, Rz, Measure]" in code or docs
- LLM maps to ProximA gate set: "Hadamard, CNOT, T, RotationZ, Measurement"

**Technical Approach:**
- Use `os.walk()` to find installed files
- Read README, API documentation, or sample code
- Send relevant content to LLM
- LLM extracts capabilities through text understanding
- No API introspection library needed - LLM reads documentation
- Generate JSON metadata structure

**Key Insight:** LLM can read backend documentation (README, API docs) and extract supported features without querying a formal API.

### Step 6.2: LLM-Generated Configuration Entry

LLM creates ProximA configuration entry by understanding existing config structure:

**Process:**
- Locate ProximA config: `~/.proxima/config/default.yaml` or `%USERPROFILE%\.proxima\config\default.yaml`
- Read existing configuration file
- Prompt LLM: "This is ProximA's configuration file: {config_yaml}. I need to add a new backend with: Name={name}, Path={path}, Executable={exe}, Capabilities={capabilities}. Generate the YAML entry to add to the 'backends:' section, following the same structure as existing entries."
- LLM analyzes existing backend entries in YAML
- LLM generates new entry matching the schema
- Append LLM-generated YAML to configuration

**Example LLM output:**
```yaml
backends:
  lret_cirq_scalability:
    name: "LRET Cirq Scalability"
    type: "lret"
    path: "/home/user/LRET/build"
    executable: "quantum_sim"
    enabled: true
    supports_noise: true
    max_qubits: 26
```

**Technical Approach:**
- Read existing YAML using `pathlib` and `open()`
- Send to LLM for structure understanding
- LLM generates conforming YAML
- Append to file using `ruamel.yaml` (preserves formatting) or simple text append
- Validate YAML syntax after modification
- No hardcoded ProximA schema - LLM infers from examples

**Key Insight:** LLM can learn ProximA's configuration schema by reading existing entries and generate conforming entries.

### Step 6.3: LLM-Managed Backend Registry

LLM registers backend by updating registry and creating wrapper:

**Process:**
- Find ProximA backend registry (backends.json or Python registry module)
- Prompt LLM: "ProximA's backend registry: {registry_content}. Add new backend: {backend_info}. Also, I need to create a Python wrapper class that ProximA can use to interact with this backend. The wrapper should: 1) Load the backend (import module or spawn process), 2) Translate ProximA circuit to backend format, 3) Execute simulation, 4) Return results. Generate: 1) Registry entry JSON, 2) Python wrapper class structure (describe methods, not full implementation)."
- LLM generates registry entry and wrapper class outline

**For wrapper class:**
- LLM describes methods needed: `__init__(config)`, `execute_circuit(circuit, params)`, `get_results()`, etc.
- Developer or code generation tool implements based on LLM's outline

**Technical Approach:**
- Read registry file
- LLM generates registry entry
- LLM outlines wrapper class structure
- Create Python file with wrapper class
- Register using ProximA's registration mechanism (entry points or direct import)
- No hardcoded ProximA protocol - LLM infers from existing wrappers

**Key Insight:** LLM can understand how existing backends are wrapped and create similar wrappers for new backends.

### Step 6.4: LLM-Based Parameter Mapping

LLM creates parameter translation between ProximA and backend:

**Process:**
- Prompt LLM: "ProximA uses these standard parameters: {proxima_params} (e.g., num_qubits, noise_level, shots, optimization_level). Backend {backend_name} uses these parameters: {backend_params} (from documentation or command-line help). Create mapping: 1) Which ProximA params map to which backend params? 2) Any transformations needed (e.g., ProximA noise_level 0.01 → backend --noise 0.01)? 3) Default values for parameters not specified? 4) Validation rules?"
- LLM generates parameter mapping specification

**Example LLM mapping:**
```json
{
  "num_qubits": {"backend_param": "-n", "transform": "direct"},
  "noise_level": {"backend_param": "--noise", "transform": "multiply_by_1"},
  "shots": {"backend_param": "--shots", "transform": "direct", "default": 1024},
  "optimization_level": {"backend_param": "--opt", "transform": "map", "mapping": {0: "none", 1: "light", 2: "aggressive"}}
}
```

**Technical Approach:**
- Extract ProximA's parameter schema
- Read backend's parameter documentation (--help output, README)
- Send both to LLM
- LLM creates mapping specification (JSON)
- Implement parameter translator using mapping
- No hardcoded parameter mappings - LLM generates based on documentation

**Key Insight:** LLM can read backend's command-line help (--help output) and ProximA's parameter docs to create accurate mappings.

---

## Phase 7: Context-Aware Runtime Execution & Parameter Mapping

**Objective:** LLM translates circuits, maps parameters, and manages execution dynamically for any backend.

### Step 7.1: LLM-Driven Circuit Translation

LLM translates ProximA circuits to any backend format:

**Process:**
- ProximA provides circuit in standard format (OpenQASM, JSON, or internal representation)
- Prompt LLM: "Translate this quantum circuit to {backend_name} format. Circuit: {circuit_representation}. Backend uses: {backend_api_summary}. Generate: 1) Backend-specific circuit construction code/commands, 2) Gate mapping (ProximA gates → backend gates), 3) Handle any gates not natively supported (decomposition needed)."
- LLM generates translation code or command-line circuit specification

**For gate decomposition:**
- If ProximA circuit has gate not supported by backend
- LLM decomposes: "Backend doesn't support Toffoli. Decompose into: CNOT + H + T + CNOT sequence"
- LLM generates decomposed circuit

**Example:**
- Input: ProximA circuit with H, CNOT, Measure gates
- LLM output for Cirq: `circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1)])`
- LLM output for C++ backend: `circuit.add_gate(H, 0); circuit.add_gate(CNOT, 0, 1); circuit.add_measurement({0, 1});`

**Technical Approach:**
- Send circuit representation to LLM with backend API description
- LLM generates code in target language (Python, C++, command-line)
- Execute generated code or command
- No hardcoded gate mapping dictionary - LLM understands gate equivalences
- LLM handles gate decompositions using quantum computing knowledge

**Key Insight:** LLM knows gate equivalences (Toffoli = controlled-controlled-X, can be decomposed into single-qubit and CNOT gates) without hardcoded decomposition rules.

### Step 7.2: LLM-Configured Execution Parameters

LLM translates user parameters to backend-specific configuration:

**Process:**
- User specifies: shots=1024, noise_level=0.01, optimization="high", parallelization="hybrid"
- Prompt LLM: "User wants to run simulation with: {user_params}. Backend {backend_name} accepts parameters: {backend_param_spec}. Generate: 1) Complete command or API call with all parameters set, 2) Any parameter transformations needed, 3) Default values for unspecified parameters."
- LLM generates configured execution command

**Example:**
- User params: {shots: 1024, noise: 0.01}
- LLM output: `./quantum_sim -n 10 -d 20 --shots 1024 --noise 0.01 --mode hybrid`
- Or Python: `simulator.run(circuit, repetitions=1024, noise_model=cirq.depolarize(0.01))`

**For complex configurations:**
- User: "high optimization" → LLM: "Backend's high optimization = -O3 flag + --optimize-gates=aggressive"
- User: "1% noise" → LLM: "Backend expects 0.01 (decimal), conversion needed"

**Technical Approach:**
- Send user parameters and backend documentation to LLM
- LLM generates complete configured command/API call
- No parameter mapping dictionary - LLM translates on the fly
- Validate LLM-generated command syntax before execution
- Execute configured command

**Key Insight:** LLM can read parameter documentation and generate correct commands without predefined mappings.

### Step 7.3: LLM-Managed Job Execution

LLM orchestrates job submission and monitoring:

**Process:**
- Execute backend command/API call using `subprocess` or Python API
- Prompt LLM: "I'm executing: {command}. This backend typically: {execution_behavior_from_docs}. Should this: 1) Run synchronously (wait for completion)? 2) Run asynchronously (submit and poll status)? 3) How to track progress? 4) How to detect completion?"
- LLM determines execution strategy
- Monitor execution based on LLM's guidance

**For long-running jobs:**
- LLM suggests: "This will take several minutes. Run in background, poll status every 10 seconds by checking output file."
- Implement polling based on LLM's suggestion

**For job cancellation:**
- User requests cancel
- LLM determines: "Send SIGTERM to process" or "Call backend's cancel API: {api_call}"
- Execute cancellation method

**Technical Approach:**
- Use `subprocess.Popen` for command-line backends
- Use Python API for Python-based backends
- LLM determines whether to wait or poll
- LLM suggests status check method (output file, log file, API call)
- No hardcoded job management - LLM adapts to backend

**Key Insight:** Different backends have different execution models (synchronous, asynchronous, batch queue). LLM determines the appropriate approach.

### Step 7.4: LLM-Parsed Result Retrieval

LLM extracts and converts results from any backend format:

**Process:**
- Backend produces results (output file, stdout, API response)
- Read results
- Prompt LLM: "Backend produced this output: {output}. Extract: 1) Measurement results (histogram of bitstrings and counts), 2) Execution time, 3) Any additional metrics (fidelity, final state, memory used), 4) Convert to ProximA standard format: {proxima_result_schema}"
- LLM parses backend output and converts to ProximA format

**Example:**
- Backend output: `00: 512 times, 11: 512 times, Time: 2.3s, Fidelity: 0.998`
- LLM extracts: `{"measurements": {"00": 512, "11": 512}, "time_ms": 2300, "fidelity": 0.998}`
- Converts to ProximA schema

**For complex outputs:**
- Backend outputs statevector in CSV file
- LLM: "Read CSV file, parse as complex numbers, format as ProximA statevector object"

**Technical Approach:**
- Read result files or capture stdout
- Send to LLM for parsing
- LLM extracts measurements, metrics, metadata
- LLM converts to ProximA's expected JSON/dict structure
- No format-specific parsers - LLM reads any format
- Return formatted results to ProximA

**Key Insight:** LLM can parse CSV, JSON, plain text, or any custom output format and extract relevant information.

---

## Phase 8: Intelligent Monitoring, Error Handling & Auto-Recovery

**Objective:** Use LLM for dynamic monitoring, intelligent error diagnosis, and adaptive recovery without hardcoded error patterns.

### Step 8.1: LLM-Assisted Real-Time Monitoring

LLM interprets process behavior and detects issues:

**Process:**
- Monitor backend process using `psutil`
- Collect metrics: CPU%, memory usage, I/O, process status
- Every 5-10 seconds, send metrics to LLM: "Process monitoring: CPU={cpu}%, Memory={mem}MB, Status={status}, Running for {duration}s. Expected behavior: {backend_behavior}. Analysis: 1) Is process healthy? 2) Any concerning patterns (high memory growth, CPU stuck at 0%)? 3) Should continue waiting or intervene?"
- LLM analyzes metrics and flags issues

**For progress detection:**
- Read backend log file or stdout
- Send log snippets to LLM: "Latest log output: {log_lines}. Previous output: {prev_lines}. Is backend making progress or stuck?"
- LLM detects: "Log shows increasing iteration count (iter 145 → iter 167) → making progress" or "Same error repeated 5 times → stuck"

**Technical Approach:**
- Use `psutil.Process()` for resource monitoring
- Implement background monitoring thread
- Periodically send metrics to LLM
- LLM detects anomalies (memory leak, hanging, thrashing)
- No hardcoded thresholds - LLM reasons about acceptable resource usage
- Display LLM's status assessment to user

**Key Insight:** Instead of hardcoding "if memory > 90% → warning", LLM considers context ("simulation of 14 qubits needs lots of memory, 80% usage is normal").

### Step 8.2: LLM-Based Error Detection & Classification

LLM diagnoses any runtime error:

**Process:**
- Detect error: non-zero exit code, stderr output, timeout, crash signal
- Gather error context: exit code, stderr, stdout, system logs, memory dump if available
- Prompt LLM: "Process failed. Exit code: {code}, stderr: {stderr}, stdout: {stdout}, signal: {signal}. System state: Memory={mem}, Disk={disk}. Diagnose: 1) Error category (crash/timeout/OOM/logic error/infrastructure)? 2) Root cause? 3) Is this user's fault, backend bug, or environment issue? 4) Severity (critical/recoverable/transient)?"
- LLM provides detailed diagnosis

**Example diagnoses:**
- Exit code 137 + high memory → LLM: "OOM kill by Linux kernel. Process exceeded memory limits."
- Stderr: "Segmentation fault" → LLM: "Backend crashed. Likely memory access bug in backend code."
- Timeout + CPU 0% → LLM: "Process hung, not consuming CPU. Likely deadlock or waiting for unavailable resource."
- Stderr: "Permission denied: /tmp/output.dat" → LLM: "Infrastructure error. Insufficient filesystem permissions."

**Technical Approach:**
- Capture all error information (exit code, signals, output)
- Send comprehensive context to LLM
- LLM classifies and diagnoses error
- No hardcoded error pattern matching
- LLM explains error in user-friendly terms
- Store diagnosis for recovery phase

**Key Insight:** LLM can diagnose obscure errors ("SIGABRT signal" → "assertion failed in backend") without error pattern database.

### Step 8.3: LLM-Guided Adaptive Recovery

LLM suggests and executes recovery strategies:

**Process:**
- After error diagnosis, prompt LLM: "Error diagnosis: {diagnosis_from_step_2}. Determine: 1) Is automatic recovery possible? 2) What recovery strategies to try (in priority order)? 3) For each strategy: action, risk level, success probability. 4) Should ask user permission for risky actions?"
- LLM generates recovery plan
- Execute recovery strategies in order

**Example recovery plans:**
- OOM error → LLM suggests:
  1. Retry with reduced parallelism (-j16 → -j4) [safe, 70% success]
  2. Retry with smaller problem size if configurable [safe, 80% success]
  3. Increase swap space (requires sudo) [risky, ask user, 90% success]
- Timeout → LLM suggests:
  1. Retry with 2× timeout [safe, 60% success]
  2. Check if progress was being made (reanalyze logs) [diagnostic]
  3. If no progress, likely infinite loop - don't retry [abort]
- Permission error → LLM suggests:
  1. Retry with different output directory (user's home) [safe, 95% success]
  2. Fix permissions with chmod (ask user) [medium risk]

**For retry logic:**
- LLM tracks retry history
- Prompt: "Previous recovery attempts: {history}. All failed. Try different approach or give up?"
- LLM suggests alternative strategies or escalates to user

**Technical Approach:**
- LLM generates recovery action list
- Validate actions for safety
- Execute safe actions automatically
- Request user permission for risky actions (sudo, file deletion)
- Retry with LLM-modified parameters
- Track attempts, let LLM decide when to stop
- No hardcoded error→recovery mapping

**Key Insight:** LLM can reason about recovery strategies contextually. Not all timeouts should retry (hanging vs slow computation), not all OOM errors can be solved by reducing parallelism.

### Step 8.4: LLM-Enhanced Logging & Diagnostics

LLM creates comprehensive diagnostic reports:

**Process:**
- When error occurs, collect: logs, system info, configurations, command history, process dumps
- Prompt LLM: "Create diagnostic report for this error: {error_summary}. Available data: {data_summary}. Generate report with: 1) Executive summary (what went wrong in 2-3 sentences), 2) Detailed error analysis, 3) Relevant log excerpts (with LLM annotations), 4) System state at failure, 5) Suggested next steps for user, 6) Information needed for bug report."
- LLM generates human-readable diagnostic report
- Save report for user review or bug reporting

**For log analysis:**
- LLM reads full log file
- LLM highlights relevant sections: "This warning at line 145 preceded the crash"
- LLM filters out noise: "These 500 lines of debug output are not relevant to the error"

**Technical Approach:**
- Use Python `logging` module for structured logs
- Collect system information using `platform`, `psutil`
- Send all data to LLM for report generation
- LLM creates Markdown-formatted diagnostic report
- No hardcoded report template - LLM tailors to error type
- Save report to file, display summary to user

**Key Insight:** LLM can synthesize information from multiple sources (logs, system state, error messages) and create coherent diagnostic narratives.

### Step 8.5: LLM-Validated Health Checks

LLM designs and evaluates health checks dynamically:

**Process:**
- Prompt LLM: "Design health check for backend {backend_name}: 1) What should be verified (executable exists, dependencies available, can execute simple command)? 2) Quick test to run (minimal circuit that completes in <5 seconds)? 3) Expected output indicating health?"
- LLM designs health check procedure
- Execute health check periodically (e.g., daily or before each simulation)
- Send results to LLM: "Health check results: {results}. Is backend healthy?"
- LLM validates and flags issues

**Example health check designed by LLM:**
1. Check executable exists and is executable: `test -x /path/to/backend`
2. Check dependencies: `ldd /path/to/backend` (Linux) or similar
3. Run minimal test: `backend -n 2 -d 1 --test-mode`
4. Expected output: "Test passed" or exit code 0
5. LLM validates: "All checks passed, backend is healthy"

**For degraded health:**
- LLM detects: "Executable exists but dependencies missing: library libfoo.so.5 not found"
- LLM suggests: "Reinstall libfoo or rebuild backend"

**Technical Approach:**
- LLM generates health check script
- Execute checks using `subprocess`
- Send results to LLM for evaluation
- LLM determines health status and suggests fixes if unhealthy
- No hardcoded health checks - LLM adapts to backend type
- Update backend status in registry based on LLM's assessment

**Key Insight:** Different backends need different health checks. LLM creates appropriate checks for compiled C++ backends, Python modules, cloud APIs, etc.

---

## Summary

All phases (1-8) are now **100% AI-driven** with:

✅ **Zero hardcoded mappings** (no dependency databases, error pattern lists, command templates)  
✅ **LLM analyzes and reasons dynamically** (reads files, interprets outputs, generates commands)  
✅ **Works with any LLM** (Gemini 2.5 Flash, GPT-4+, Claude 3.5+, Ollama models)  
✅ **Adapts to any backend** (quantum simulators, compilers, frameworks - no backend-specific code)  
✅ **Self-improving through context** (LLM learns from previous attempts in conversation)

**Implementation Note:** Each step involves prompting the LLM with specific questions and context, receiving structured responses (usually JSON), and executing actions based on LLM's reasoning. The system is driven by LLM intelligence, not hardcoded logic.
