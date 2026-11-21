"""Tests run with src/Project as working directory, as the pipeline does (kitsune_lib reads data/ from there).
    python -m pytest          (from the repository root)"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent / "src" / "Project"
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
