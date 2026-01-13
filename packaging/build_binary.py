#!/usr/bin/env python
"""
Build script for creating Proxima binary distributions.

Supports:
- PyInstaller single-file executable
- Directory bundle with dependencies
- Cross-platform builds (Windows, macOS, Linux)

Usage:
    python build_binary.py [--onefile] [--clean] [--debug]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PACKAGING_DIR = PROJECT_ROOT / "packaging"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def check_pyinstaller() -> bool:
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        return False


def install_pyinstaller() -> bool:
    """Install PyInstaller if not present."""
    print("Installing PyInstaller...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller>=6.0.0"],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyInstaller: {e}")
        return False


def clean_build_dirs():
    """Remove previous build artifacts."""
    print("Cleaning build directories...")
    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed: {dir_path}")


def get_platform_suffix() -> str:
    """Get platform-specific suffix for executable name."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return f"win-{machine}"
    elif system == "darwin":
        if machine == "arm64":
            return "macos-arm64"
        return "macos-x64"
    else:
        return f"linux-{machine}"


def build_with_spec(debug: bool = False) -> Path:
    """Build using the PyInstaller spec file."""
    spec_file = PACKAGING_DIR / "proxima.spec"
    
    if not spec_file.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_file}")
    
    print(f"Building with spec file: {spec_file}")
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]
    
    if debug:
        cmd.append("--debug=all")
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=not debug,
        text=True,
    )
    
    if result.returncode != 0:
        if result.stderr:
            print(f"Build errors:\n{result.stderr}")
        raise RuntimeError("PyInstaller build failed")
    
    # Find the output executable
    if platform.system() == "Windows":
        exe_path = DIST_DIR / "proxima.exe"
    else:
        exe_path = DIST_DIR / "proxima"
    
    if not exe_path.exists():
        raise FileNotFoundError(f"Expected output not found: {exe_path}")
    
    return exe_path


def build_onefile(debug: bool = False) -> Path:
    """Build single-file executable without spec file."""
    print("Building single-file executable...")
    
    entry_point = SRC_DIR / "proxima" / "__main__.py"
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--clean",
        "--noconfirm",
        "--name", "proxima",
        "--paths", str(SRC_DIR),
        # Add hidden imports
        "--hidden-import", "pydantic",
        "--hidden-import", "typer",
        "--hidden-import", "rich",
        "--hidden-import", "structlog",
        "--hidden-import", "textual",
        # Add data files
        "--add-data", f"{PROJECT_ROOT / 'configs'}{os.pathsep}configs",
        str(entry_point),
    ]
    
    if debug:
        cmd.append("--debug=all")
    
    print(f"Running: {' '.join(cmd[:10])}...")
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=not debug,
        text=True,
    )
    
    if result.returncode != 0:
        if result.stderr:
            print(f"Build errors:\n{result.stderr}")
        raise RuntimeError("PyInstaller build failed")
    
    # Find the output
    if platform.system() == "Windows":
        exe_path = DIST_DIR / "proxima.exe"
    else:
        exe_path = DIST_DIR / "proxima"
    
    return exe_path


def rename_with_platform(exe_path: Path) -> Path:
    """Rename executable with platform suffix."""
    suffix = get_platform_suffix()
    
    if platform.system() == "Windows":
        new_name = f"proxima-{suffix}.exe"
    else:
        new_name = f"proxima-{suffix}"
    
    new_path = exe_path.parent / new_name
    shutil.move(exe_path, new_path)
    print(f"Renamed to: {new_path}")
    
    return new_path


def create_checksum(exe_path: Path) -> Path:
    """Create SHA256 checksum file."""
    import hashlib
    
    print(f"Creating checksum for: {exe_path.name}")
    
    sha256_hash = hashlib.sha256()
    with open(exe_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    checksum = sha256_hash.hexdigest()
    checksum_file = exe_path.with_suffix(exe_path.suffix + ".sha256")
    
    with open(checksum_file, "w") as f:
        f.write(f"{checksum}  {exe_path.name}\n")
    
    print(f"Checksum: {checksum}")
    return checksum_file


def verify_build(exe_path: Path) -> bool:
    """Verify the built executable works."""
    print(f"Verifying build: {exe_path}")
    
    try:
        result = subprocess.run(
            [str(exe_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode == 0:
            print(f"Build verified: {result.stdout.strip()}")
            return True
        else:
            print(f"Verification failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Verification timed out")
        return False
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build Proxima binary distribution")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build without spec file (simple onefile mode)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories before building",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug mode enabled",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help="Don't add platform suffix to executable name",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip build verification",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Proxima Binary Build")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print()
    
    # Ensure PyInstaller is available
    if not check_pyinstaller():
        if not install_pyinstaller():
            print("ERROR: Could not install PyInstaller")
            return 1
    
    # Clean if requested
    if args.clean:
        clean_build_dirs()
    
    # Build
    try:
        if args.onefile:
            exe_path = build_onefile(debug=args.debug)
        else:
            exe_path = build_with_spec(debug=args.debug)
    except Exception as e:
        print(f"Build failed: {e}")
        return 1
    
    print(f"\nBuild successful: {exe_path}")
    print(f"Size: {exe_path.stat().st_size / (1024 * 1024):.2f} MB")
    
    # Rename with platform suffix
    if not args.no_rename:
        exe_path = rename_with_platform(exe_path)
    
    # Create checksum
    create_checksum(exe_path)
    
    # Verify
    if not args.no_verify:
        if not verify_build(exe_path):
            print("WARNING: Build verification failed")
            return 1
    
    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"Output: {exe_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
