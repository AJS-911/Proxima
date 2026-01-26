#!/usr/bin/env python
"""
Build configuration for Proxima to handle Visual Studio version compatibility.

This module automatically detects the installed Visual Studio version and
configures the build system to work with any version of Visual Studio
(2015, 2017, 2019, 2022, and future versions).

It solves the qiskit-aer and other C++ extension compilation issues
on Windows by:
1. Auto-detecting available MSVC compiler versions
2. Setting up environment variables for compatibility
3. Providing fallback options for pre-built wheels
4. Using compatible package versions
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple


class VisualStudioDetector:
    """Detect installed Visual Studio versions and MSVC compilers."""

    def __init__(self):
        self.vs_versions = {}
        self.msvc_versions = {}
        self.detect()

    def detect(self):
        """Detect all installed Visual Studio versions."""
        if platform.system() != "Windows":
            return

        # Try to detect via vswhere.exe
        self._detect_via_vswhere()

        # Try registry detection
        self._detect_via_registry()

        # Try environment variables
        self._detect_via_env()

    def _detect_via_vswhere(self):
        """Detect Visual Studio via vswhere.exe."""
        vswhere_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
            r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
        ]

        for vswhere_path in vswhere_paths:
            if Path(vswhere_path).exists():
                try:
                    result = subprocess.run(
                        [vswhere_path, "-all", "-format", "json"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        import json
                        installations = json.loads(result.stdout)
                        for inst in installations:
                            version = inst.get("displayVersion", "")
                            install_path = inst.get("installationPath", "")
                            if version and install_path:
                                self.vs_versions[version] = install_path
                except Exception as e:
                    print(f"Warning: vswhere detection failed: {e}")
                break

    def _detect_via_registry(self):
        """Detect Visual Studio via Windows registry."""
        try:
            import winreg
        except ImportError:
            return

        try:
            # Check for Visual Studio installation paths
            vs_key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, vs_key_path) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    subkey_name = winreg.EnumKey(key, i)
                    if "Visual Studio" in subkey_name or "MSVC" in subkey_name:
                        try:
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                install_path = winreg.QueryValueEx(
                                    subkey, "InstallLocation"
                                )[0]
                                if install_path and Path(install_path).exists():
                                    self.vs_versions[subkey_name] = install_path
                        except WindowsError:
                            pass
        except Exception as e:
            print(f"Warning: Registry detection failed: {e}")

    def _detect_via_env(self):
        """Detect Visual Studio via environment variables."""
        vs_env_vars = [
            "VSINSTALLDIR",
            "VisualStudioVersion",
            "VCToolsInstallDir",
            "WindowsSDKVersion",
        ]

        for var in vs_env_vars:
            value = os.environ.get(var)
            if value:
                self.vs_versions[var] = value

    def get_best_version(self) -> Optional[str]:
        """Get the best available Visual Studio version."""
        if not self.vs_versions:
            return None

        # Prefer newer versions
        version_priority = {
            "17": 5,  # VS 2022
            "16": 4,  # VS 2019
            "15": 3,  # VS 2017
            "14": 2,  # VS 2015
        }

        best_version = None
        best_priority = -1

        for version_str in self.vs_versions.keys():
            for priority_version, priority in version_priority.items():
                if priority_version in version_str and priority > best_priority:
                    best_version = version_str
                    best_priority = priority

        return best_version or next(iter(self.vs_versions.keys()), None)

    def get_msvc_version_tuple(self) -> Optional[Tuple[int, int]]:
        """Get MSVC compiler version as (major, minor) tuple."""
        best = self.get_best_version()
        if not best:
            return None

        # Map Visual Studio versions to MSVC versions
        version_map = {
            "17": (193,),  # VS 2022 (MSVC 193.x)
            "16": (192,),  # VS 2019 (MSVC 192.x)
            "15": (191,),  # VS 2017 (MSVC 191.x)
            "14": (190,),  # VS 2015 (MSVC 190.x)
        }

        for vs_ver, msvc_ver in version_map.items():
            if vs_ver in best:
                return msvc_ver

        return None

    def setup_build_environment(self):
        """Setup environment variables for the build process."""
        best_version = self.get_best_version()

        if not best_version:
            print("Warning: No Visual Studio installation detected!")
            print("Build may fail. Please install Visual Studio with C++ workload.")
            return False

        print(f"Detected Visual Studio: {best_version}")

        # Set environment variables for compatibility
        if "17" in best_version or "2022" in best_version:
            os.environ["DISTUTILS_USE_SDK"] = "1"
            os.environ["MSSdk"] = "1"
        elif "16" in best_version or "2019" in best_version:
            os.environ["DISTUTILS_USE_SDK"] = "1"
        elif "15" in best_version or "2017" in best_version:
            pass  # Use default settings for VS 2017

        return True


def setup_environment_for_build():
    """Setup the build environment."""
    if platform.system() == "Windows":
        detector = VisualStudioDetector()
        detector.setup_build_environment()
        return detector
    return None


def get_compatible_dependencies() -> list[str]:
    """Get a list of compatible dependencies that work across VS versions."""
    # These versions have good pre-built wheel support on PyPI
    # and don't require specific compiler versions
    dependencies = [
        "typer>=0.9.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "structlog>=23.2.0",
        "anyio>=4.0.0",
        "psutil>=5.9.0",
        "httpx>=0.25.0",
        "pandas>=2.1.0",
        "openpyxl>=3.1.0",
        "pyyaml>=6.0",
        "transitions>=0.9.0",
        "keyring>=24.0.0",
        "rich>=13.0.0",
        # Quantum backends - using versions with good wheel support
        "cirq>=1.3.0",
        "qiskit>=0.45.0",
        # Use latest qiskit-aer which has better pre-built wheel support
        "qiskit-aer>=0.14.0",  # Updated for better compatibility
    ]
    return dependencies


def suggest_installation_command():
    """Suggest the best installation command."""
    print("\n" + "=" * 70)
    print("INSTALLATION GUIDANCE")
    print("=" * 70)
    print("\nTo install Proxima with maximum compatibility:")
    print("\n1. Using pre-built wheels (RECOMMENDED):")
    print("   pip install --upgrade pip setuptools wheel")
    print("   pip install -e .[all]")
    print("\n2. If wheels aren't available, skip C++ extensions initially:")
    print("   pip install --only-binary :all: -e .[all]")
    print("\n3. If you still encounter issues:")
    print("   pip install -e . --no-binary qiskit-aer")
    print("   # Then install without qiskit-aer initially")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Setup build environment when invoked
    detector = setup_environment_for_build()
    suggest_installation_command()

    if detector:
        print(f"Detected Visual Studio versions: {detector.vs_versions}")
        print(f"Best version selected: {detector.get_best_version()}")
