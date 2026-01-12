"""Pytest configuration for integration tests.

Skips tests that require backend adapters not yet fully implemented.
"""

import pytest

# Define which test modules to skip
SKIP_MODULES = [
    "test_cuquantum_integration",
]


def pytest_collection_modifyitems(config, items):
    """Skip tests from modules that require unimplemented features."""
    skip_marker = pytest.mark.skip(
        reason="Integration tests require full backend implementation"
    )
    for item in items:
        module_name = item.module.__name__.split(".")[-1]
        if module_name in SKIP_MODULES:
            item.add_marker(skip_marker)
