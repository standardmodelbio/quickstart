"""Pytest configuration. Add scripts/ to path so prep_mimic_demo_data can be imported."""
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent / "scripts"
if _scripts.exists() and str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
