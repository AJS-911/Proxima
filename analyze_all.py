import re
import os

def analyze_file(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
    methods = re.findall(r'^\s+(?:async\s+)?def\s+(\w+)', content, re.MULTILINE)
    top_funcs = re.findall(r'^(?:async\s+)?def\s+(\w+)', content, re.MULTILINE)
    
    return {
        'lines': len(lines),
        'classes': classes,
        'methods': list(set(methods)),
        'top_funcs': top_funcs,
        'content': content
    }

base_path = r'C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima\src\proxima\core'

# SESSION.PY
print('='*60)
print('SESSION.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'session.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Key methods: {info["methods"][:30]}')
content = info['content']

print('\nFeature Detection:')
print(f'  sqlite mentions: {"sqlite" in content.lower()}')
print(f'  json file storage: {"json" in content.lower() and "file" in content.lower()}')
print(f'  export/import: {"export" in content.lower() and "import" in content.lower()}')
print(f'  concurrent mentions: {"concurrent" in content.lower() or "lock" in content.lower()}')
print(f'  recovery mentions: {"recover" in content.lower() or "crash" in content.lower()}')
print(f'  persistence: {"persist" in content.lower() or "save" in content.lower()}')

# STATE.PY
print('\n' + '='*60)
print('STATE.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'state.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'All methods: {info["methods"]}')
content = info['content']

print('\nFeature Detection:')
print(f'  persistence: {"persist" in content.lower() or "save" in content.lower()}')
print(f'  cleanup/abort: {"cleanup" in content.lower() or "abort" in content.lower()}')
print(f'  transition validation: {"transition" in content.lower() and "valid" in content.lower()}')
print(f'  state machine: {"transition" in content.lower()}')

# AGENT_INTERPRETER.PY
print('\n' + '='*60)
print('AGENT_INTERPRETER.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'agent_interpreter.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Top functions: {info["top_funcs"]}')
print(f'Key methods count: {len(info["methods"])}')
content = info['content']

print('\nFeature Detection:')
print(f'  agent.md parsing: {"agent" in content.lower() and (".md" in content.lower() or "markdown" in content.lower())}')
print(f'  command validation: {"valid" in content.lower() and "command" in content.lower()}')
print(f'  error recovery: {"recover" in content.lower() or "retry" in content.lower()}')
print(f'  error handling: {"error" in content.lower() or "exception" in content.lower()}')

# EXECUTOR.PY
print('\n' + '='*60)
print('EXECUTOR.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'executor.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Methods: {info["methods"]}')
print(f'Top functions: {info["top_funcs"]}')

# PLANNER.PY
print('\n' + '='*60)
print('PLANNER.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'planner.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Key methods: {info["methods"][:25]}')
print(f'Top functions: {info["top_funcs"]}')

# RUNNER.PY
print('\n' + '='*60)
print('RUNNER.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'runner.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Methods: {info["methods"]}')
print(f'Top functions: {info["top_funcs"]}')
