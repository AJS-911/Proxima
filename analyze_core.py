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
        'methods': methods,
        'top_funcs': top_funcs,
        'content': content
    }

base_path = r'C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima\src\proxima\core'

# Analyze pipeline.py
print('='*60)
print('PIPELINE.PY ANALYSIS')
print('='*60)
info = analyze_file(os.path.join(base_path, 'pipeline.py'))
print(f'Lines: {info["lines"]}')
print(f'Classes: {info["classes"]}')
print(f'Top functions: {info["top_funcs"]}')
content = info['content']

# Feature detection
print('\nFeature Detection:')
print(f'  pause/resume mentions: {"pause" in content.lower() and "resume" in content.lower()}')
print(f'  pause method exists: {bool(re.search(r"def\s+pause", content))}')
print(f'  resume method exists: {bool(re.search(r"def\s+resume", content))}')
print(f'  rollback mentions: {"rollback" in content.lower()}')
print(f'  rollback method exists: {bool(re.search(r"def\s+rollback", content))}')
print(f'  checkpoint mentions: {"checkpoint" in content.lower()}')
print(f'  checkpoint methods: {bool(re.search(r"def\s+.*checkpoint", content))}')
print(f'  DAG/visualization: {"dag" in content.lower() or "visualiz" in content.lower()}')
print(f'  distributed/parallel: {"distributed" in content.lower() or "parallel" in content.lower()}')
