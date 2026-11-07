"""Shared Artsy tool-agent instructions, independent of the evaluation framework."""

from drawing.documentation import DOCUMENTATION_LOCATION

from pathlib import Path

from .config import LIMITS, MODEL_CONTEXT_WINDOW

ROOT = Path(__file__).resolve().parents[1]


def drawing_system_prompt(token_limit: int | None = MODEL_CONTEXT_WINDOW, token_limit_type: str = "all",
                          max_calls: int | None = LIMITS['maxCalls']) -> str:
    """Share drawing instructions without exposing execution budgets to the model.

    Budget arguments remain accepted for callers and recorded trajectory configs.
    Limits are enforced by the harness rather than included in instructions.
    """
    return (
        "Fulfill the drawing request by writing /workspace/program.py. "
        "Use the installed ArtCanvas v2 API (from artcanvas import ArtCanvas) to save output.png at 1920×1200. "
        "Use bash commands to write, run, and fix your program. "
        "Write the program and run its preview in separate bash calls on separate turns. "
        "Use the preview command to receive both execution feedback and the actual PNG image. "
        "The final program is executed again in a fresh directory with no other files. "
        "A text answer or a pre-existing PNG does not count. "
        'If you cannot fulfill the request, write only refusal.json containing {"reason":"..."} '
        "with a concise reason and do not create program.py. "
        "\n\n"
        + DOCUMENTATION_LOCATION
    )


# Version 1 denotes historical unversioned recordings; version 2 records the
# host turn limit and the countdown emitted before every model call.
TRAJECTORY_CONTRACT_VERSION = 2
# The interactive/evaluation harness now uses named drawing tools. Historical
# bash recordings and the training importer keep their original version above.
DRAWING_TOOLS_CONTRACT_VERSION = 3
SFT_MAX_TURNS = 12


def _drawing_tool(name: str, description: str, argument: str | None = None, detail: str = "") -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": (
            {argument: {"type": "string", "description": detail}} if argument else {}),
            "required": [argument] if argument else [], "additionalProperties": False}}}


DRAWING_TOOLS = [
    _drawing_tool("update_drawing", "Create or replace the drawing with a complete ArtCanvas v2 Python program.",
                  "source", "Complete Python source saving a 1920×1200 output.png, at most 64 KiB."),
    _drawing_tool("read_drawing", "Read the current drawing's Python source."),
    _drawing_tool("preview_drawing", "Render the current drawing and return execution feedback and its PNG preview."),
    _drawing_tool("submit_drawing", "Finish the drawing and submit its current source for independent final validation."),
    _drawing_tool("refuse_drawing", "Decline the drawing request with a reason and end without submitting artwork.",
                  "reason", "Concise reason for declining, at most 2048 UTF-8 bytes."),
]


def drawing_tools_prompt() -> str:
    """Instructions for the host harness's structured drawing tools."""
    return (
        "Fulfill the drawing request using the drawing tools. "
        "Use update_drawing(source) to create or replace the complete Python program. "
        "Use the installed ArtCanvas v2 API (from artcanvas import ArtCanvas) to save output.png at 1920×1200. "
        "Use read_drawing() to inspect the current source if needed. "
        "After writing, call preview_drawing() on a separate turn to receive execution "
        "feedback and the actual PNG image. Revise the source and preview again as needed. "
        "Call submit_drawing() when the drawing is ready. The harness renders the final "
        "source again in a fresh directory with no other files. "
        "A text answer or a pre-existing PNG does not count. "
        "If you cannot fulfill the request, call refuse_drawing(reason) with a concise "
        "reason; this ends the request without submitting artwork. "
        "Every response must include at least one drawing tool call. "
        "Tools run in order; do not call more tools after submission or refusal.\n\n"
        + DOCUMENTATION_LOCATION
    )


def turn_limit_instruction(max_turns: int | None) -> str:
    if max_turns is None:
        return ""
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError("max_turns must be positive or None")
    return (f"\n\nYou have {max_turns} model turns to complete this task. "
            "Each response uses one turn. Submit your final output by the last turn.")


def turn_countdown(turn: int, max_turns: int) -> str:
    turn_limit_instruction(max_turns)
    if type(turn) is not int or not 1 <= turn <= max_turns:
        raise ValueError("turn must be within max_turns")
    return (f"Turn {turn} of {max_turns}." +
            (" This is your final turn; submit your final output now."
             if turn == max_turns else ""))


def host_user_prompt(system_prompt: str, task: str, max_turns: int | None = None) -> str:
    return ("System instructions:\n" + system_prompt + turn_limit_instruction(max_turns)
            + "\n\nTask:\n" + task)
