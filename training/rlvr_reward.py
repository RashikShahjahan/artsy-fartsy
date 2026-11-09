"""Deterministic, penalty-only verifiers for the SFT -> RLVR stage.

The best possible reward is zero. Failed final rendering, malformed tool calls,
and failed executions incur independent negative penalties; successes never
grant credit or dilute earlier failures.
"""

RLVR_ALGORITHM = "tinker-cookbook-rlvr-v1"
RLVR_REWARD_VERSION = "artsy-rlvr-verifiers-v2"

IMAGE_WEIGHT = 0.75
TOOL_WEIGHT = 0.125
EXECUTION_WEIGHT = 0.125


def verified_reward(*, submitted: bool, render_valid: bool,
                    valid_tool_calls: int, invalid_tool_calls: int,
                    execution_attempts: int, execution_errors: int) -> dict[str, float]:
    """Penalize verified failures; a clean submitted result earns exactly zero.

    A missing/invalid submitted image is one failure. Every malformed tool call
    and every failed shell execution or preview incurs another penalty. An
    episode with no tool calls incurs one tool-protocol penalty. Counts, rather
    than failure fractions, prevent successful trivial commands from diluting
    prior errors. Non-penalty metrics remain available for diagnostics.
    """
    values = (valid_tool_calls, invalid_tool_calls, execution_attempts, execution_errors)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("RLVR evidence counts must be nonnegative integers")
    if execution_errors > execution_attempts:
        raise ValueError("Execution errors cannot exceed attempts")

    image_verified = float(bool(submitted and render_valid))
    tool_attempts = valid_tool_calls + invalid_tool_calls
    tool_protocol = valid_tool_calls / tool_attempts if tool_attempts else 0.0
    execution_success = ((execution_attempts - execution_errors) / execution_attempts
                         if execution_attempts else 0.0)

    image_penalty = -IMAGE_WEIGHT if not image_verified else 0.0
    tool_penalty = -TOOL_WEIGHT * (invalid_tool_calls if tool_attempts else 1)
    execution_penalty = -EXECUTION_WEIGHT * execution_errors
    reward = image_penalty + tool_penalty + execution_penalty
    return {
        "image_verified": image_verified,
        "tool_protocol": tool_protocol,
        "execution_success": execution_success,
        "image_penalty": image_penalty,
        "tool_penalty": tool_penalty,
        "execution_penalty": execution_penalty,
        "reward": reward,
    }
