"""Location and provenance of pycairo's installed API reference."""

import hashlib
from importlib.metadata import distribution
import json

DOCUMENTATION_LOCATION = (
    'Official pycairo API reference (searchable with rg): /opt/artsy/pycairo-api.pyi. '
    'Before writing code, search the installed API reference for the operations you plan to use. '
)


def api_documentation() -> str:
    """Record the installed package version and its complete reference file."""
    package = distribution('pycairo')
    reference = package.locate_file('cairo/__init__.pyi')
    return json.dumps({
        'package': 'pycairo', 'version': package.version,
        'files': {'cairo/__init__.pyi': hashlib.sha256(reference.read_bytes()).hexdigest()},
    }, sort_keys=True)
