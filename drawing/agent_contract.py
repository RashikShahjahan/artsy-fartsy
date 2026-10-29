"""Shared Artsy tool-agent instructions, independent of the evaluation framework."""

from pathlib import Path

from .config import LIMITS, MODEL_CONTEXT_WINDOW

ROOT = Path(__file__).resolve().parents[1]


def drawing_system_prompt(token_limit: int | None = MODEL_CONTEXT_WINDOW, token_limit_type: str = "all") -> str:
    """Share drawing instructions while describing the caller's actual budget."""
    documentation = (ROOT / "docs/artcanvas.md").read_text()
    token_description = "generated output" if token_limit_type == "output" else "total input and output"
    token_budget = f" and {token_limit} {token_description} tokens" if token_limit is not None else ""
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
        f"You have at most {LIMITS['maxCalls']} model turns{token_budget}.\n\n"
        + documentation
    )
