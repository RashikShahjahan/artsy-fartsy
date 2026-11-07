"""The ArtCanvas v2 reference and reproducible API provenance."""

import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "artcanvas.md"
IMPLEMENTATION = ROOT / "runtime" / "artcanvas.py"

# Small enough to provide directly instead of spending a bash turn discovering docs.
DOCUMENTATION_LOCATION = (
    "Use ArtCanvas v2, not raw pycairo. Its complete API reference follows:\n\n"
    + REFERENCE.read_text(encoding="utf-8")
)


def api_documentation() -> str:
    """Hash ArtCanvas's public guide and implementation, plus its Cairo backend."""
    return json.dumps({
        "api": "artcanvas-v2",
        "backend": "pycairo",
        "backend_version": distribution("pycairo").version,
        "files": {
            "docs/artcanvas.md": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            "runtime/artcanvas.py": hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest(),
        },
    }, sort_keys=True)
