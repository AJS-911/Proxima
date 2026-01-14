"""Script to fix lret.py ExecutionResult constructor."""

# Read the file
with open('src/proxima/backends/lret.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 0: Remove the bad decorator @wrap_backend_exception("lret")
content = content.replace('    @wrap_backend_exception("lret")\n    def execute(', '    def execute(')

# Fix 1: Replace the ExecutionResult constructor call in execute method
old_execute_result = '''        # Build ExecutionResult
        return ExecutionResult(
            result_type=ResultType.COUNTS if normalized.counts else ResultType.STATE_VECTOR,
            counts=normalized.counts or None,
            statevector=normalized.statevector,
            metadata={
                "backend": "lret",
                "shots": shots,
                "execution_time_ms": execution_time_ms,
                "normalized": True,
                "format": normalized.format.value,
                **normalized.metadata,
            },
        )'''

new_execute_result = '''        # Build ExecutionResult with proper signature
        data = {}
        if normalized.counts:
            data["counts"] = normalized.counts
        if normalized.statevector is not None:
            data["statevector"] = normalized.statevector
        if normalized.probabilities:
            data["probabilities"] = normalized.probabilities
        
        result_type = ResultType.COUNTS if normalized.counts else ResultType.STATEVECTOR
        
        return ExecutionResult(
            backend="lret",
            simulator_type=SimulatorType.CUSTOM,
            execution_time_ms=execution_time_ms,
            qubit_count=num_qubits,
            shot_count=shots if shots > 0 else None,
            result_type=result_type,
            data=data,
            metadata={
                "normalized": True,
                "format": normalized.format.value,
                **normalized.metadata,
            },
            raw_result=result,
        )'''

content = content.replace(old_execute_result, new_execute_result)

# Fix 2: Update estimate_resources to work with mock mode
# Find and replace the specific if statement
old_estimate = '        if not self.is_available():'
new_estimate = '''        # Allow estimation even when LRET is not installed if using mock mode
        if not self.is_available() and not self._use_mock:'''

# Only replace the one in estimate_resources, not in _get_lret_module
# The estimate_resources method starts after the comment about resources
# We need to find the specific occurrence

# Split by the estimate_resources method definition
parts = content.split('def estimate_resources(self, circuit: Any) -> ResourceEstimate:')
if len(parts) == 2:
    before_method = parts[0]
    after_def = parts[1]
    
    # Only replace in the after_def part, and only the first occurrence
    after_def = after_def.replace(old_estimate, new_estimate, 1)
    
    content = before_method + 'def estimate_resources(self, circuit: Any) -> ResourceEstimate:' + after_def

# Write the file
with open('src/proxima/backends/lret.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed lret.py')
