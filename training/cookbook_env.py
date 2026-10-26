"""Tinker Cookbook rollout environment for Artsy (experimental).

This module intentionally does not invoke Inspect. Each episode owns one isolated
Docker container. Model sampling remains on Tinker. Construct an ArtsyEnv with
the same Qwen renderer used for SFT.
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import uuid
from pathlib import Path

from PIL import Image
import tinker
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

from drawing.config import IMAGE_SIZE
from inference.server import parse_tools
from inference.tinker_backend import TinkerBackend

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_TURNS = 16
TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command in the drawing workspace.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]}}}


async def docker(*args, stdin=None, timeout=30):
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise
    return proc.returncode, stdout, stderr


class ArtsyEnv(Env):
    """Single-use, multi-turn drawing task with a sandboxed bash tool."""

    def __init__(self, prompt: str, seed: int, renderer, image_processor,
                 image_size=IMAGE_SIZE, *, image_name="artsy-eval"):
        self.prompt, self.seed = prompt, seed
        self.renderer, self.image_processor = renderer, image_processor
        self.image_size, self.image_name = image_size, image_name
        self.container = None
        self.messages = []
        self.turns = 0
        self.submitted = False
        self.last_image = None
        self.commands = []
        self.tools = [TOOL]

    def _render(self):
        # Reuse the production Qwen tool/vision rendering protocol.
        backend = object.__new__(TinkerBackend)
        backend.renderer = self.renderer
        backend.image_processor = self.image_processor
        backend.image_size = self.image_size
        return backend.render(self.messages, self.tools)

    async def initial_observation(self):
        self.container = "artsy-rl-" + uuid.uuid4().hex
        code, _, stderr = await docker(
            "run", "-d", "--name", self.container, "--network", "none",
            "--memory", "2g", "--cpus", "2", self.image_name,
            "tail", "-f", "/dev/null")
        if code:
            raise RuntimeError("Unable to start isolated episode: " + stderr.decode()[-512:])
        api = (ROOT / "docs/artcanvas.md").read_text()
        system = (ROOT / "evaluation/mini.yaml").read_text()
        import yaml
        settings = yaml.safe_load(system)
        self.messages = [
            {"role": "system", "content": settings["agent"]["system_template"]},
            {"role": "user", "content": (
                "System instructions:\nFulfill the drawing request by writing /workspace/program.py. "
                "Use ArtCanvas to save output.png at 1920x1200. Write code and "
                "run its preview in separate tool calls. If unsafe, write only "
                'refusal.json with {"reason":"..."} and no program.py. '
                "The final code is scored in a fresh directory.\n\n" + api +
                f"\nTask:\n{self.prompt}\n\nPreview with: python /opt/artsy/render.py "
                f"--seed {self.seed} --output /workspace/preview --preview < /workspace/program.py")}]
        return self._render(), self.renderer.get_stop_sequences()

    async def step(self, action, *, extra=None):
        if self.container is None:
            raise RuntimeError("Call initial_observation first")
        self.turns += 1
        text = self.renderer.tokenizer.decode(action, skip_special_tokens=False)
        try:
            _, calls = parse_tools(text, self.tools)
        except (ValueError, SyntaxError):
            calls = []
        if not calls:
            self.messages.append({"role": "assistant", "content": text})
            self.messages.append({"role": "user", "content":
                                  'Every response must contain a bash tool call with {"command":"..."}.'})
        else:
            # Preserve the wire-format tool calls as context on the next turn.
            self.messages.append({"role": "assistant", "content": "",
                                  "tool_calls": [{**c, "function": {**c["function"],
                                      "arguments": json.loads(c["function"]["arguments"])}}
                                      for c in calls]})
            for call in calls:
                command = json.loads(call["function"]["arguments"])["command"]
                self.commands.append(command)
                code, out, err = await docker(
                    "exec", "-i", "-w", "/workspace", self.container,
                    "timeout", "--signal=KILL", "10", "bash", "-c", command,
                    timeout=14)
                result = f"Return code: {code}\n" + out.decode(errors="replace")[:8192]
                result += err.decode(errors="replace")[:8192]
                parts = [{"type": "text", "text": result}]
                # Read the preview image independently from the untrusted output.
                if "--preview" in command and "render.py" in command and code == 0:
                    status, png, _ = await docker("cp",
                        f"{self.container}:/workspace/preview/image.png", "-",
                        timeout=10)
                    # Docker cp - emits TAR: do not treat it as raw PNG.
                    if status == 0:
                        import tarfile
                        try:
                            with tarfile.open(fileobj=io.BytesIO(png)) as archive:
                                member = next(m for m in archive if m.isfile()
                                              and m.name.endswith("image.png"))
                                raw = archive.extractfile(member).read(8 * 1024 * 1024 + 1)
                            if len(raw) <= 8 * 1024 * 1024:
                                with Image.open(io.BytesIO(raw)) as im:
                                    im.verify()
                                with Image.open(io.BytesIO(raw)) as im:
                                    parts.append({"type": "image", "image": im.convert("RGB")})
                        except (OSError, ValueError, StopIteration):
                            pass
                self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                      "name": "bash", "content": parts})
                if command.strip() == SUBMIT and code == 0:
                    self.submitted = True
                    break
        done = self.submitted or self.turns >= MAX_TURNS
        # A separate trusted final renderer and judge must provide reward.
        return StepResult(reward=0.0, episode_done=done,
                          next_observation=self._render() if not done else tinker.ModelInput.from_ints([]),
                          next_stop_condition=self.renderer.get_stop_sequences() if not done else [],
                          metrics={"tool_calls": float(len(self.commands))},
                          logs={"commands": self.commands.copy()})

    async def cleanup(self):
        if self.container:
            await docker("rm", "-f", self.container, timeout=15)
            self.container = None
