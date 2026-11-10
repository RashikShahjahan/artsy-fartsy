"""Deterministic, penalty-only verifiers for the SFT -> RLVR stage.

The best possible reward is zero. Failed final rendering, malformed tool calls,
and failed executions incur independent negative penalties; successes never
grant credit or dilute earlier failures.
"""

RLVR_ALGORITHM = "tinker-cookbook-rlvr-v1"
RLVR_REWARD_VERSION = "artsy-rlvr-verifiers-v3"

IMAGE_WEIGHT = 1.0
ERROR_WEIGHT = 0.025
MAX_PENALIZED_ERRORS = 4


def verified_reward(*, submitted: bool, render_valid: bool,
                    valid_tool_calls: int, invalid_tool_calls: int,
                    execution_attempts: int, execution_errors: int) -> dict[str, float]:
    """Penalize verified failures; a clean submitted result earns exactly zero.

    A missing/invalid submitted image is one failure. Every malformed tool call
    and every failed shell execution or preview incurs another penalty. An
    episode with no tool calls incurs one tool-protocol penalty. Protocol and
    execution penalties are capped independently. Successful commands do not
    dilute prior errors. Non-penalty metrics remain available for diagnostics.
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
    protocol_errors = invalid_tool_calls if tool_attempts else 1
    tool_penalty = -ERROR_WEIGHT * min(protocol_errors, MAX_PENALIZED_ERRORS)
    execution_penalty = -ERROR_WEIGHT * min(execution_errors, MAX_PENALIZED_ERRORS)
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
