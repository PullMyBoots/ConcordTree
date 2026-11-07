"""Load the isolated Web dependencies without changing scientific packages."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'webdeps'))
runpy.run_path(str(root / 'web/server.py'), run_name='__main__')
