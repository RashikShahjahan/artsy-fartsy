"""Deterministic, judge-free rewards for the SFT -> RLVR training stage.

Evidence comes from the agent's sandbox tool outcomes and a trusted final
render in a fresh sandbox. Incomplete episodes can earn limited process credit
to help bootstrap submission, but never receive the final-image bonus.
"""

RLVR_ALGORITHM = "tinker-cookbook-rlvr-v1"
RLVR_REWARD_VERSION = "artsy-rlvr-verifiers-v1"

IMAGE_WEIGHT = 0.75
TOOL_WEIGHT = 0.125
EXECUTION_WEIGHT = 0.125


def verified_reward(*, submitted: bool, render_valid: bool,
                    valid_tool_calls: int, invalid_tool_calls: int,
                    execution_attempts: int, execution_errors: int) -> dict[str, float]:
    """Score only observed verifiable events, never text or judge preference.

    A valid tool call has the exact advertised schema. Execution success
    includes command return codes and explicit failed preview feedback. Tool
    protocol and execution credit are capped at 0.25 without a verified
    submitted final PNG; zero tool attempts never count as correct.
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
    reward = (IMAGE_WEIGHT * image_verified + TOOL_WEIGHT * tool_protocol
              + EXECUTION_WEIGHT * execution_success)
    return {"image_verified": image_verified, "tool_protocol": tool_protocol,
            "execution_success": execution_success, "reward": reward}
