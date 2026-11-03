"""Shared Artsy tool-agent instructions, independent of the evaluation framework."""

from pathlib import Path

from .config import LIMITS, MODEL_CONTEXT_WINDOW

ROOT = Path(__file__).resolve().parents[1]


def drawing_system_prompt(token_limit: int | None = MODEL_CONTEXT_WINDOW, token_limit_type: str = "all",
                          max_calls: int | None = LIMITS['maxCalls']) -> str:
    """Share drawing instructions without exposing execution budgets to the model.

    Budget arguments remain accepted for callers and recorded trajectory configs.
    Limits are enforced by the harness rather than included in instructions.
    """
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
        "\n\n"
        + documentation
    )


# Version 1 denotes historical unversioned recordings; version 2 records the
# host turn limit and the countdown emitted before every model call.
TRAJECTORY_CONTRACT_VERSION = 2


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
