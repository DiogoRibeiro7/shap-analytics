"""
Pytest configuration for benchmarks.

Shared fixtures and configuration for all benchmark tests.
"""

import pytest


def pytest_configure(config):
    """Add custom markers for benchmarks."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "api: marks tests that require API server")
    config.addinivalue_line("markers", "memory: marks tests that profile memory usage")


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test names."""
    for item in items:
        if "10k" in item.nodeid.lower() or "large" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
