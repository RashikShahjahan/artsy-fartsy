"""Shared Artsy tool-agent instructions, independent of the evaluation framework."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def drawing_system_prompt() -> str:
    """Keep the model-facing drawing task identical for training and evaluation."""
    documentation = (ROOT / "docs/artcanvas.md").read_text()
    return (
        "Fulfill the drawing request by writing /workspace/program.py. "
        "Use the installed ArtCanvas module to save output.png at 1920×1200. "
        "Use bash commands to write, run, and fix your program. "
        "Write the program and run its preview in separate bash calls on separate turns. "
        "Use the preview command to receive both execution feedback and the actual PNG image. "
        "The final program is executed again in a fresh directory with no other files. "
        "A text answer or a pre-existing PNG does not count. "
        'If you cannot fulfill the request, write only refusal.json containing {"reason":"..."} '
        "with a concise reason and do not create program.py. "
        "You have at most sixteen model turns and 32768 total input and output tokens.\n\n"
        + documentation
    )
