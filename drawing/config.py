"""Shared limits for drawing episodes and local rendering."""

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
IMAGE_SIZE = {"shortest_edge": 256 * 256, "longest_edge": 1024 * 1024}

# Keep the names used by the persisted environment config.
LIMITS = {
    "maxCalls": 16,
    "maxAttempts": 3,
    "renderSeconds": 10,
    "episodeMs": 300_000,
    "sourceBytes": 64 * 1024,
}
