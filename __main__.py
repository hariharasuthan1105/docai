"""
Entry point when package is run directly via `python -m docai` or `python .`.
"""

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
for _p in [_parent_dir, _curr_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from docai.cli import main
except ImportError:
    from cli import main

if __name__ == "__main__":
    main()
