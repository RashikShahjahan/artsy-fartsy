"""Shared limits for drawing episodes and local rendering."""

MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROVIDER = "qwen-local"

# Keep the names used by the persisted environment config and TS callers.
LIMITS = {
    "maxCalls": 16,
    "maxAttempts": 3,
    "maxTurns": 16,
    "renderSeconds": 10,
    "episodeMs": 300_000,
    "sourceBytes": 64 * 1024,
}
