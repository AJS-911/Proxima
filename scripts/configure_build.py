#!/usr/bin/env python
"""
Visual Studio Version Compatibility Configuration Script.

This script configures the build environment to work with any version
of Visual Studio (2015, 2017, 2019, 2022, and future versions).

It solves the common issue where quantum computing packages like qiskit-aer
require specific Visual Studio versions by:

1. Automatically detecting installed Visual Studio versions
2. Configuring environment variables for the detected compiler
3. Setting up distutils/setuptools to use the correct MSVC toolchain
4. Providing fallback options when compilation is not possible

Usage:
    python scripts/configure_build.py
    pip install -e .[all]

For future-proofing, this script uses dynamic detection rather than
hardcoding specific VS version numbers.
"""

import os
import sys
import platform
import subprocess
import json
from pathlib import Path
from typing import Optional


class VSVersionError(Exception):
    """Raised when Visual Studio version cannot be determined."""
    pass


class VisualStudioConfigurator:
    """Configure Visual Studio for Python package builds."""

    # Map Visual Studio version numbers to their internal version and MSVC version
    VS_VERSION_MAP = {
        # Format: (install_version, generator_name, msvc_version, year)
        "17": ("17.0", "Visual Studio 17 2022", "v143", 2022),
        "16": ("16.0", "Visual Studio 16 2019", "v142", 2019),
        "15": ("15.0", "Visual Studio 15 2017", "v141", 2017),
        "14": ("14.0", "Visual Studio 14 2015", "v140", 2015),
    }

    def __init__(self):
        self.detected_vs = None
        self.vs_path = None
        self.vs_version_info = None
        self.is_windows = platform.system() == "Windows"

    def detect_visual_studio(self) -> bool:
        """
        Detect installed Visual Studio versions.
        Returns True if at least one version is found.
        """
        if not self.is_windows:
            print("Not running on Windows - Visual Studio detection skipped")
            return True

        print("Detecting Visual Studio installations...")

        # Method 1: Use vswhere.exe (most reliable)
        if self._detect_via_vswhere():
            return True

        # Method 2: Check environment variables
        if self._detect_via_environment():
            return True

        # Method 3: Check common installation paths
        if self._detect_via_paths():
            return True

        print("Warning: No Visual Studio installation detected!")
        return False

    def _detect_via_vswhere(self) -> bool:
        """Detect VS using vswhere.exe (available since VS 2017)."""
        vswhere_paths = [
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        ]

        for vswhere_path in vswhere_paths:
            if vswhere_path.exists():
                try:
                    # Get all installed VS versions
                    result = subprocess.run(
                        [
                            str(vswhere_path),
                            "-all",
                            "-prerelease",
                            "-products", "*",
                            "-format", "json",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        installations = json.loads(result.stdout)
                        if installations:
                            # Get the newest installation
                            newest = max(
                                installations,
                                key=lambda x: x.get("installationVersion", "0.0.0")
                            )
                            self.detected_vs = newest.get("displayVersion", "Unknown")
                            self.vs_path = newest.get("installationPath", "")

                            # Determine VS version info
                            for ver_prefix, info in self.VS_VERSION_MAP.items():
                                if self.detected_vs.startswith(ver_prefix):
                                    self.vs_version_info = info
                                    break

                            # If version is newer than known, use the newest known config
                            if not self.vs_version_info:
                                # Future-proofing: assume new versions are compatible with VS 2022 config
                                self.vs_version_info = self.VS_VERSION_MAP["17"]
                                print(f"Detected future VS version {self.detected_vs}, using VS 2022 configuration")

                            print(f"Detected Visual Studio {self.detected_vs} at {self.vs_path}")
                            return True
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                    print(f"vswhere detection failed: {e}")

        return False

    def _detect_via_environment(self) -> bool:
        """Detect VS using environment variables."""
        # Check for VS environment variables (set when running from VS Developer Command Prompt)
        vsinstalldir = os.environ.get("VSINSTALLDIR", "")
        vs_version = os.environ.get("VisualStudioVersion", "")

        if vsinstalldir and vs_version:
            self.detected_vs = vs_version
            self.vs_path = vsinstalldir

            for ver_prefix, info in self.VS_VERSION_MAP.items():
                if vs_version.startswith(ver_prefix):
                    self.vs_version_info = info
                    break

            if not self.vs_version_info:
                self.vs_version_info = self.VS_VERSION_MAP["17"]

            print(f"Detected Visual Studio {self.detected_vs} from environment")
            return True

        return False

    def _detect_via_paths(self) -> bool:
        """Detect VS by checking common installation paths."""
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        ]

        vs_paths = []
        for pf in program_files:
            for year in [2022, 2019, 2017, 2015]:
                for edition in ["Enterprise", "Professional", "Community", "BuildTools"]:
                    path = Path(pf) / "Microsoft Visual Studio" / str(year) / edition
                    if path.exists():
                        vs_paths.append((year, path))

        if vs_paths:
            # Use the newest version found
            year, path = max(vs_paths, key=lambda x: x[0])
            self.detected_vs = str(year)
            self.vs_path = str(path)

            ver_prefix = {2022: "17", 2019: "16", 2017: "15", 2015: "14"}.get(year, "17")
            self.vs_version_info = self.VS_VERSION_MAP.get(ver_prefix, self.VS_VERSION_MAP["17"])

            print(f"Detected Visual Studio {year} at {path}")
            return True

        return False

    def configure_environment(self):
        """Configure environment variables for the build process."""
        if not self.is_windows:
            return

        print("\nConfiguring build environment...")

        # Set up environment for any Visual Studio version
        if self.vs_version_info:
            install_ver, generator, toolset, year = self.vs_version_info

            # These environment variables help setuptools/distutils find the right compiler
            os.environ["DISTUTILS_USE_SDK"] = "1"
            os.environ["MSSdk"] = "1"

            # Set the CMAKE generator for packages that use cmake
            os.environ["CMAKE_GENERATOR"] = generator

            # For packages that check toolset version
            os.environ["VS_TOOLSET_VERSION"] = toolset

            print(f"  CMAKE_GENERATOR = {generator}")
            print(f"  VS_TOOLSET_VERSION = {toolset}")
            print(f"  DISTUTILS_USE_SDK = 1")

        # Additional settings for better compatibility
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        # Prefer pre-built wheels
        os.environ["PIP_PREFER_BINARY"] = "1"
        os.environ["PIP_ONLY_BINARY"] = ":all:"

        print("\nEnvironment configured for build compatibility.")

    def configure_pip(self):
        """Configure pip for optimal package installation."""
        print("\nConfiguring pip settings...")

        # Create or update pip configuration
        pip_config_dir = Path.home() / "pip"
        pip_config_dir.mkdir(exist_ok=True)
        pip_config_file = pip_config_dir / "pip.ini"

        # Read existing config if present
        existing_config = {}
        if pip_config_file.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(pip_config_file)
                for section in config.sections():
                    existing_config[section] = dict(config.items(section))
            except Exception:
                pass

        # Add our compatibility settings
        if "install" not in existing_config:
            existing_config["install"] = {}

        # Prefer binary wheels to avoid compilation issues
        existing_config["install"]["prefer-binary"] = "true"

        # Write the config
        try:
            import configparser
            config = configparser.ConfigParser()
            for section, options in existing_config.items():
                if section not in config:
                    config.add_section(section)
                for key, value in options.items():
                    config.set(section, key, value)

            with open(pip_config_file, 'w') as f:
                config.write(f)
            print(f"  Updated pip config: {pip_config_file}")
        except Exception as e:
            print(f"  Could not update pip config: {e}")

    def create_setup_cfg(self):
        """Create a setup.cfg file with Visual Studio compatibility settings."""
        project_root = Path(__file__).parent.parent
        setup_cfg_path = project_root / "setup.cfg"

        content = """\
# Setup configuration for Visual Studio compatibility
# Auto-generated by configure_build.py

[bdist_wheel]
# Use the stable ABI for maximum compatibility
# py3-none-any is preferred when there are no C extensions

[build_ext]
# Compiler settings for C extensions
# Uses the detected Visual Studio version automatically

[options]
# Prefer pre-built wheels for dependencies
install_requires = file: requirements.txt
"""

        with open(setup_cfg_path, 'w') as f:
            f.write(content)
        print(f"\nCreated setup.cfg at {setup_cfg_path}")

    def run(self):
        """Run the full configuration process."""
        print("=" * 70)
        print("Visual Studio Build Configuration for Proxima")
        print("=" * 70)
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version}")
        print()

        # Detect Visual Studio
        if self.is_windows:
            if not self.detect_visual_studio():
                print("\n" + "!" * 70)
                print("WARNING: Visual Studio not detected!")
                print("You may need to install Visual Studio with C++ workload.")
                print("Download from: https://visualstudio.microsoft.com/downloads/")
                print("!" * 70)
                print("\nContinuing with pre-built wheel strategy...")

        # Configure environment
        self.configure_environment()

        # Configure pip
        self.configure_pip()

        print("\n" + "=" * 70)
        print("Configuration complete!")
        print("=" * 70)
        print("\nNow run:")
        print("  pip install --upgrade pip setuptools wheel")
        print("  pip install -e .[all]")
        print("\nIf you encounter issues, try:")
        print("  pip install --only-binary :all: -e .[all]")
        print("=" * 70)


def main():
    configurator = VisualStudioConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
