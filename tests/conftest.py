"""Shared pytest setup: put the repo root on sys.path so `import src...` works.

pytest loads this automatically for every test under tests/ (including subfolders).
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
