import sys
import os.path as path

# Add croco/ to sys.path so that "import models" finds croco/models/
HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_PATH = path.normpath(path.join(HERE_PATH, '../../croco'))
if CROCO_PATH not in sys.path:
    sys.path.insert(0, CROCO_PATH)
