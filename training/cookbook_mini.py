"""Tinker Cookbook adapter for Artsy's mini-SWE bash drawing harness.

Keep the rollout protocol identical to SFT and Inspect: a persistent Docker
workspace, one bash(command) tool, optional artsy-preview image observations,
and an explicit artsy-submit or native mini completion.
"""

import asyncio
import base64
import io
import json
import subprocess

from PIL import Image

from drawing.mini_agent import CONTRACT_VERSION, TOOLS, system_prompt
from drawing.mini_session import DrawingSession, READ_SOURCE
from drawing.model_profiles import resolve_model
from drawing.vision import prepare_image
from training.cookbook_env import DockerSession, _docker, encoder, MAX_PNG


def mini_tool_spec():
    """Use the tool schema recorded by mini-SWE SFT, not a similar hand-written one."""
    return TOOLS[0].get("function", TOOLS[0])


def mini_initial_messages(spec, model, renderer, max_length, revision):
    prefix = encoder(model, renderer, max_length, revision).renderer.create_conversation_prefix_with_tools(
        [mini_tool_spec()], system_prompt=system_prompt(spec["seed"]))
    return [*prefix, {"role": "user", "content": spec["prompt"]}]


def _observation(content, model, renderer, max_length, revision):
    """Translate trusted mini preview parts to Cookbook's native image representation."""
    converted = []
    for part in content:
        if part["type"] == "text":
            converted.append(part)
            continue
        if part["type"] != "image_url":
            raise ValueError("Unknown mini-SWE observation type")
        url = part["image_url"]["url"]
        prefix = "data:image/png;base64,"
        if not isinstance(url, str) or not url.startswith(prefix) or len(url) > len(prefix) + 4 * ((MAX_PNG + 2) // 3):
            raise ValueError("Invalid or oversized mini-SWE preview")
        png = base64.b64decode(url[len(prefix):], validate=True)
        if len(png) > MAX_PNG:
            raise ValueError("Mini-SWE preview exceeds 8 MiB")
        with Image.open(io.BytesIO(png)) as image:
            if image.format != "PNG" or not 1 <= image.width <= 4096 or not 1 <= image.height <= 4096:
                raise ValueError("Invalid mini-SWE PNG")
            rgb = image.convert("RGB")
        processor = encoder(model, renderer, max_length, revision).image_processor
        converted.append({"type": "image", "image": prepare_image(
            rgb, processor, vision=resolve_model(model)["vision"])})
    return converted if len(converted) > 1 else converted[0]["text"]


class MiniDockerSession(DockerSession):
    """Share Docker capacity and trusted final-image grading with legacy rollouts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory_contract_version = CONTRACT_VERSION
        self.drawing = DrawingSession(self._shell, self._snapshot)

    async def _shell(self, command):
        await self.start()
        # The untrusted command is an argv value for bash *inside* the isolated
        # container, never interpolated into a shell on the training host.
        try:
            result = await _docker(
                ["exec", "-i", "-w", "/workspace", "-e", f"ARTSY_SEED={self.spec['seed']}",
                 self.container, "timeout", "--signal=KILL", "65", "bash", "-lc", command],
                timeout=70)
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "output": "Drawing command timed out"}
        return {"returncode": result.returncode,
                "output": (result.stdout + b"\n" + result.stderr).decode(errors="replace")}

    async def _snapshot(self, entrypoint=None):
        if not self.container:
            raise ValueError("No active mini-SWE workspace")
        try:
            result = await _docker(
                ["exec", "-w", "/workspace", self.container, "python", "-c", READ_SOURCE]
                + ([entrypoint] if entrypoint else []), timeout=12)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("Mini-SWE submission snapshot timed out") from error
        if result.returncode or len(result.stdout) > 512 * 1024:
            raise ValueError("Cannot snapshot mini-SWE workspace: " + result.stderr.decode(errors="replace")[-2048:])
        return json.loads(result.stdout)

    async def execute_bash(self, command):
        if not isinstance(command, str) or not command.strip():
            return "Return code: 1\nExpected a bash command.", False
        if self.submitted:
            return "Return code: 1\nThis request has already ended.", True
        result = await self.drawing.execute(command)
        self.tool_calls.append({"name": "bash", "arguments": {"command": command}})
        self.previews.extend(result.get("previewEvents", []))
        self.submitted = self.drawing.submitted
        observation = await asyncio.to_thread(
            _observation, result["content"], self.model, self.renderer, self.max_length, self.revision)
        return observation, self.submitted

    async def _read_submission(self):
        # The mini-SWE submit command captured a validated multi-file bundle.
        # Only that committed source, not an arbitrary workspace file, is scored.
        return (self.drawing.source if self.submitted else ""), None


class MiniBashTool:
    def __init__(self, session):
        self.session = session
        definition = mini_tool_spec()
        self.name = definition["name"]
        self.description = definition["description"]
        self.parameters_schema = definition["parameters"]

    def to_spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters_schema}

    async def run(self, input):
        from tinker_cookbook.tool_use.tools import simple_tool_result
        arguments = input.arguments
        if not isinstance(arguments, dict) or set(arguments) != {"command"} or not isinstance(arguments["command"], str):
            observation, done = "Return code: 1\nExpected bash(command: string).", False
        else:
            observation, done = await self.session.execute_bash(arguments["command"])
        return simple_tool_result(observation, call_id=input.call_id or "",
                                  name="bash", should_stop=done)
