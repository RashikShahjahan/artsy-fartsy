#!/usr/bin/env python3
"""Generate browsable per-symbol Markdown from the installed pycairo .pyi.

Usage: python scripts/generate_pycairo_markdown.py SOURCE.pyi OUTPUT_DIR
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import re
import shutil
import sys


FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)


def signature(node):
    """Preserve overloads and full annotated method signatures."""
    clone = copy.deepcopy(node)
    clone.body = [ast.Pass()]
    return "\n".join(ast.unparse(clone).splitlines()[:-1])


def markdown_docstring(value):
    """Convert common reStructuredText roles and fields without dropping prose."""
    doc = inspect.cleandoc(value)
    doc = re.sub(r":(?:class|meth|func|attr|mod|obj|data|const|exc|ref|type):\u0060(?:[^\u0060<>]+\s*<)?([^\u0060<>]+)>?\u0060",
                 lambda m: chr(96) + m.group(1) + chr(96), doc)
    doc = re.sub(r"(?m)^:param\s+([\w*]+):\s*(.*)$", r"- **\1**: \2", doc)
    doc = re.sub(r"(?m)^:type\s+([\w*]+):\s*(.*)$", r"  Type of **\1**: \2", doc)
    doc = re.sub(r"(?m)^:returns?:\s*(.*)$", r"**Returns:** \1", doc)
    doc = re.sub(r"(?m)^:rtype:\s*(.*)$", r"**Return type:** \1", doc)
    doc = re.sub(r"(?m)^:raises?\s+([^:]+):\s*(.*)$", r"**Raises \1:** \2", doc)
    for directive, label in (("versionadded", "Added in version"),
                             ("versionchanged", "Changed in version"),
                             ("deprecated", "Deprecated in version")):
        doc = re.sub(r"(?m)^\.\. " + directive + r"::\s*(.*)$",
                     lambda m: "*" + label + " " + m.group(1) + ".*", doc)
    doc = re.sub(r"(?m)^\.\. note::\s*(.*)$", r"**Note:** \1", doc)
    doc = re.sub(r"(?m)^\.\. warning::\s*(.*)$", r"**Warning:** \1", doc)
    doc = re.sub(r"(?m)(?<!:)::\s*$", ":", doc)
    return doc.strip()


def attributes(nodes):
    """Preserve constant declarations and their adjacent descriptive strings."""
    result = []
    for i, node in enumerate(nodes):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        result += [chr(96) * 3 + "python", ast.unparse(node), chr(96) * 3, ""]
        if i + 1 < len(nodes):
            nxt = nodes[i + 1]
            if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
                result += [markdown_docstring(nxt.value.value), ""]
    return result


def write_methods(nodes, folder, prefix):
    groups = {}
    for node in nodes:
        if isinstance(node, FUNCTION):
            groups.setdefault(node.name, []).append(node)
    links = []
    for name, variants in sorted(groups.items()):
        page = [f"# {prefix}.{name}", ""]
        for variant in variants:
            page += [chr(96) * 3 + "python", signature(variant), chr(96) * 3, ""]
        docs = list(dict.fromkeys(markdown_docstring(doc)
                                  for variant in variants if (doc := ast.get_docstring(variant))))
        for doc in docs:
            page += [doc, ""]
        if not docs:
            page += ["No description in the installed stub; consult the official reference.", ""]
        (folder / f"{name}.md").write_text("\n".join(page), encoding="utf-8")
        links.append("- [" + chr(96) + name + chr(96) + f"]({name}.md)")
    return links


def generate(source: Path, destination: Path):
    raw = source.read_text(encoding="utf-8")
    tree = ast.parse(raw, filename=str(source))
    public = [node for node in tree.body
              if isinstance(node, ast.ClassDef) and not node.name.startswith("_")]
    if not public:
        raise ValueError("No public classes in pycairo stub")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("Destination must be a directory, not a file or symlink")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    bt = chr(96)
    index = ["# pycairo API reference", "",
             "Generated from the pycairo type stub installed in this environment.",
             "Each method page contains full signatures, overloads, and docstrings.", "",
             "## Navigation", "",
             "Use ls to browse and cat to read full method pages (not single-line grep).", "",
             bt * 3 + "bash", "ls /opt/artsy/docs/pycairo/",
             "cat /opt/artsy/docs/pycairo/Context/arc.md", bt * 3,
             "", "## Classes", ""]
    total = 0
    for node in sorted(public, key=lambda n: n.name):
        folder = destination / node.name
        folder.mkdir()
        links = write_methods(node.body, folder, "cairo." + node.name)
        page = ["# cairo." + node.name, ""]
        if doc := ast.get_docstring(node):
            page += [markdown_docstring(doc), ""]
        if node.bases:
            page += ["Inherits from " + ", ".join(ast.unparse(base) for base in node.bases) + ".", ""]
        attrs = attributes(node.body)
        if attrs:
            page += ["## Attributes and constants", "", *attrs]
        page += ["## Methods", "", *links, ""]
        (folder / "README.md").write_text("\n".join(page), encoding="utf-8")
        index.append(f"- [{bt}cairo.{node.name}{bt}]({node.name}/README.md)")
        total += len(links)
    module = destination / "module"
    module.mkdir()
    links = write_methods(tree.body, module, "cairo")
    page = ["# cairo module", "", "## Functions", "", *links, "",
            "## Constants and aliases", "", *attributes(tree.body)]
    (module / "README.md").write_text("\n".join(page), encoding="utf-8")
    total += len(links)
    index += ["", "- [Module functions and constants](module/README.md)", "",
              "## Provenance", "",
              "Source SHA-256: " + bt + hashlib.sha256(raw.encode()).hexdigest() + bt, "",
              "Official reference: https://pycairo.readthedocs.io/en/latest/reference/", "",
              "Pycairo is licensed under LGPL-2.1-only OR MPL-1.1; see upstream license.", ""]
    (destination / "README.md").write_text("\n".join(index), encoding="utf-8")
    return {"classes": len(public), "methods": total}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: generate_pycairo_markdown.py SOURCE.pyi OUTPUT_DIR")
    result = generate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Generated {result['classes']} classes and {result['methods']} methods/functions")
