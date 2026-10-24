"""Live tool-call and vision smoke test for Artsy's native Tinker bridge.

Start `make serve` in another terminal, then run
`uv run python -m evaluation.check_tinker`. Uses hosted Tinker sampling credits;
Docker is not required.
"""

import argparse
import base64
from io import BytesIO
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .harness import (BASE_MODEL, LOCAL_MODEL_URL, ROOT, _model_api_key,
                      check_native_bridge, model_url)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("ARTSY_MODEL", BASE_MODEL))
    args = parser.parse_args()
    if model_url().rstrip("/") != LOCAL_MODEL_URL:
        parser.error("Native Tinker requires the local bridge: remove ARTSY_MODEL_URL "
                     "from .env (and your shell), then start `make serve`")
    check_native_bridge(args.model)

    with OpenAI(base_url=model_url(), api_key=_model_api_key(),
                timeout=120, max_retries=0) as client:
        # Mini SWE needs this function-call shape to execute bash commands.
        tool_reply = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Call the bash tool with command echo READY."}],
            tools=[{"type": "function", "function": {
                "name": "bash", "description": "Execute a shell command",
                "parameters": {"type": "object", "properties": {
                    "command": {"type": "string"}}, "required": ["command"]}}}],
            tool_choice={"type": "function", "function": {"name": "bash"}},
            max_tokens=128, temperature=0,
        )
        calls = tool_reply.choices[0].message.tool_calls or []
        if not calls or calls[0].function.name != "bash":
            raise RuntimeError("Tinker did not return an OpenAI bash tool call")
        arguments = json.loads(calls[0].function.arguments)
        if not isinstance(arguments.get("command"), str):
            raise RuntimeError("Tinker returned a tool call without a command string")
        print("Tool calling: OK")

        # Artsy sends its PNG previews as inline OpenAI image_url blocks.
        png = BytesIO()
        Image.new("RGB", (64, 64), (255, 0, 0)).save(png, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()
        vision_reply = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "What is the dominant color of this image? Reply with one word."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            max_tokens=64, temperature=0,
        )
        answer = vision_reply.choices[0].message.content or ""
        if "red" not in answer.lower():
            raise RuntimeError(f"Tinker failed the PNG vision check: {answer!r}")
        print("PNG vision: OK")
    print("Native Tinker bridge passed both prerequisites for the Artsy Mini SWE agent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
