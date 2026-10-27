"""Keep source-checkout tests away from installed camera states and accounts."""
import os
from pathlib import Path
import tempfile

_temporary = tempfile.TemporaryDirectory(prefix='pyscript-tests-', ignore_cleanup_errors=True)
_root = Path(_temporary.name)
os.environ['CCTV_WEB_AUTH_PATH'] = str(_root/'users.json')
os.environ['CCTV_BACKGROUND_STATE_PATH'] = str(_root/'camera-states.json')
os.environ['SATUN_PROJECT_DIR'] = str(_root/'no-installed-satun')
