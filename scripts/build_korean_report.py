"""Compatibility entry point: generate only the current full report."""
import sys
from build_full_korean_report import main

if __name__ == '__main__':
    if '--partial' not in sys.argv:
        sys.argv.append('--partial')
    main()
