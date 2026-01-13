# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for building Proxima as a standalone executable.

Usage:
    pyinstaller proxima.spec --clean

This creates a single-file executable with all dependencies bundled.
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the project root directory
PROJECT_ROOT = Path(SPECPATH).parent
SRC_DIR = PROJECT_ROOT / "src"

# Analysis configuration
block_cipher = None

# Collect all proxima submodules
proxima_submodules = collect_submodules('proxima')

# Collect data files (templates, configs, etc.)
datas = [
    # Default configuration files
    (str(PROJECT_ROOT / 'configs'), 'configs'),
    # Template files for exporters
    (str(SRC_DIR / 'proxima' / 'data' / 'templates'), 'proxima/data/templates'),
    # Resource files
    (str(SRC_DIR / 'proxima' / 'resources' / 'prompts'), 'proxima/resources/prompts'),
]

# Filter out non-existent paths
datas = [(src, dst) for src, dst in datas if os.path.exists(src)]

# Hidden imports that PyInstaller might miss
hidden_imports = [
    # Core dependencies
    'pydantic',
    'pydantic_settings',
    'typer',
    'click',
    'rich',
    'structlog',
    'yaml',
    'toml',
    
    # Quantum backends (optional, may not be installed)
    'qiskit',
    'qiskit_aer',
    'cirq',
    
    # Export dependencies
    'openpyxl',
    'jinja2',
    
    # TUI dependencies
    'textual',
    'textual.app',
    'textual.widgets',
    'textual.screen',
    
    # Other
    'asyncio',
    'httpx',
    'aiofiles',
]

# Add all proxima submodules
hidden_imports.extend(proxima_submodules)

# Exclude unnecessary modules to reduce size
excludes = [
    'tkinter',
    'matplotlib',
    'numpy.testing',
    'pytest',
    'sphinx',
    'IPython',
    'jupyter',
    'notebook',
]

a = Analysis(
    [str(SRC_DIR / 'proxima' / '__main__.py')],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries
a.datas = list(set(a.datas))

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='proxima',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Use UPX compression if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / 'packaging' / 'icon.ico') if (PROJECT_ROOT / 'packaging' / 'icon.ico').exists() else None,
)
