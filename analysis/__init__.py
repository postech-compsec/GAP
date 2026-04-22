"""Shared analysis package setup.

Every `python -m analysis.*` entry point imports this module first. Use that
hook to redirect Matplotlib's config/cache directory into the repo so the
artifact does not emit warnings on hosts where `$HOME/.config` is not
writable.
"""

from pathlib import Path
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MPLCONFIGDIR = _PROJECT_ROOT / ".cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
