#!/usr/bin/env python
"""Shim setup.py for backwards compatibility with older tooling.

This file exists to support older pip versions and tools that don't
fully support PEP 517/518 builds with pyproject.toml.

All configuration is defined in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
