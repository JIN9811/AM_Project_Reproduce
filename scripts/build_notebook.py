"""Compatibility entry point: generate only the current full-study notebook."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('build_full_notebook.py')), run_name='__main__')
