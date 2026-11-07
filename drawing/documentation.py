"""Location and provenance of pycairo's installed API reference."""

import hashlib
from importlib.metadata import distribution
import json

DOCUMENTATION_LOCATION = (
    'Browsable pycairo Markdown API reference: /opt/artsy/docs/pycairo/README.md. '
    'Before writing code, navigate the class folders and use cat to read complete method pages, '
    'including parameters and overloads. The source type stub is /opt/artsy/pycairo-api.pyi. '
)


def api_documentation() -> str:
    """Record the installed package version and its complete reference file."""
    package = distribution('pycairo')
    reference = package.locate_file('cairo/__init__.pyi')
    return json.dumps({
        'package': 'pycairo', 'version': package.version,
        'files': {'cairo/__init__.pyi': hashlib.sha256(reference.read_bytes()).hexdigest()},
    }, sort_keys=True)
