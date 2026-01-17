import os
import re
import ast
from collections import defaultdict

core_path = r"C:\Users\dell\Pictures\intern\ProximA\Pseudo-Proxima\src\proxima\core"
files = ["agent_interpreter.py", "executor.py", "pipeline.py", "planner.py", "session.py", "state.py"]

def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    pass_count = len(re.findall(r"^\s+pass\s*$", content, re.MULTILINE))
    ellipsis_count = len(re.findall(r"^\s+\.\.\.\s*$", content, re.MULTILINE))
    not_impl_count = len(re.findall(r"raise NotImplementedError", content))
    todo_count = len(re.findall(r"#\s*(TODO|FIXME|XXX|HACK)", content, re.IGNORECASE))
    todos = re.findall(r"#\s*(TODO|FIXME|XXX|HACK)[:\s]*(.*)", content, re.IGNORECASE)
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "Syntax error in file"}
    
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
    
    class_methods = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_methods[node.name].append(item.name)
    
    stubs = []
    implemented = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            is_stub = False
            
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            
            if len(body) == 0:
                is_stub = True
            elif len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    is_stub = True
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value == ...:
                    is_stub = True
                elif isinstance(stmt, ast.Raise) and hasattr(stmt, "exc") and stmt.exc:
                    if hasattr(stmt.exc, "func") and hasattr(stmt.exc.func, "id"):
                        if stmt.exc.func.id == "NotImplementedError":
                            is_stub = True
            
            if is_stub:
                stubs.append(node.name)
            else:
                implemented.append(node.name)
    
    return {
        "lines": len(content.split("\n")),
        "classes": classes,
        "class_count": len(classes),
        "functions": functions,
        "function_count": len(functions),
        "class_methods": dict(class_methods),
        "pass_count": pass_count,
        "ellipsis_count": ellipsis_count,
        "not_impl_count": not_impl_count,
        "todo_count": todo_count,
        "todos": todos,
        "stubs": stubs,
        "stub_count": len(stubs),
        "implemented": implemented,
        "implemented_count": len(implemented),
    }

for f in files:
    fp = os.path.join(core_path, f)
    if os.path.exists(fp):
        result = analyze_file(fp)
        total = result["function_count"]
        impl = result["implemented_count"]
        stub = result["stub_count"]
        pct = (impl / total * 100) if total > 0 else 0
        
        print(f"=== {f} ===")
        print(f"Lines: {result['lines']}")
        print(f"Classes: {result['classes']}")
        print(f"Total functions/methods: {total}")
        print(f"Implemented: {impl}")
        print(f"Stubs/placeholders: {stub}")
        print(f"Completion: {pct:.1f}%")
        print(f"pass statements: {result['pass_count']}")
        print(f"... ellipsis: {result['ellipsis_count']}")
        print(f"NotImplementedError: {result['not_impl_count']}")
        print(f"TODO/FIXME comments: {result['todo_count']}")
        if result["todos"]:
            print(f"TODOs found: {result['todos']}")
        if result["stubs"]:
            print(f"Stub functions: {result['stubs']}")
        print()
    else:
        print(f"{f}: FILE NOT FOUND")
        print()
