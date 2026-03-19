"""Shared test data — loaded once per process.

Works with both pytest and unittest:
  - pytest: session-scoped fixture
  - unittest: import DATA_SNAPSHOT directly
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

from web.server import compute_data, DATA

# Compute once at import time (shared across all tests)
compute_data()
DATA_SNAPSHOT = dict(DATA)

# pytest fixture (if pytest is available)
try:
    import pytest

    @pytest.fixture(scope="session")
    def data():
        return DATA_SNAPSHOT
except ImportError:
    pass
