"""ArtCanvas Docker environment for Tinker Cookbook multi-turn RL.

Training uses Cookbook's native sampling/trajectory loop; Inspect remains the
independent evaluation harness. Both expose the same Mini SWE bash protocol.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from PIL import Image

from drawing.config import IMAGE_SIZE, MODEL_ID, MODEL_CONTEXT_WINDOW
from drawing.agent_contract import drawing_system_prompt
from drawing.judges import JUDGE_MAX_TOKENS, correctness_judge_prompt, parse_correctness_judgment
from training.data import ROOT
from training.sft import create_encoder

IMAGE = "artsy-rl-runtime"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_PNG = 8 * 1024 * 1024
SOURCE_BYTES = 64 * 1024
NO_TOOL_REMINDER = 'Every response must contain a bash tool call with {"command":"..."}.'

_session_limit: asyncio.Semaphore | None = None


def configure_concurrency(max_samples: int) -> None:
    """Limit concurrent long-lived Docker workspaces across all prompt groups."""
    global _session_limit
    _session_limit = asyncio.Semaphore(max_samples)


def _run(argv: list[str], *, stdin: bytes | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=False)


async def _docker(argv: list[str], *, stdin: bytes | None = None, timeout: float = 30):
    return await asyncio.to_thread(_run, ["docker", *argv], stdin=stdin, timeout=timeout)


@lru_cache(maxsize=1)
def encoder():
    # Cached per process; weights live on Tinker, not on this machine.
    return create_encoder(MODEL_ID, MODEL_CONTEXT_WINDOW)


@lru_cache(maxsize=1)
def agent_settings():
    import yaml
    return yaml.safe_load((ROOT / "evaluation/mini.yaml").read_text())


def initial_messages(spec: dict, tool_specs: list[dict]) -> list[dict]:
    """Use the same system/instance framing and API documentation as Inspect."""
    settings = agent_settings()
    prefix = encoder().renderer.create_conversation_prefix_with_tools(
        tool_specs, system_prompt=settings["agent"]["system_template"])
    prompt = (f'{spec["prompt"]}\n\nPreview with: '
              f'python /opt/artsy/render.py --seed {spec["seed"]} '
              '--output /workspace/preview --preview < /workspace/program.py')
    return [*prefix, {"role": "user",
                      "content": f"System instructions:\n{drawing_system_prompt(token_limit=None)}\n\nTask:\n{prompt}"}]


def _preview_observation(stdout: bytes, stderr: bytes, returncode: int) -> tuple[list[dict] | str, dict | None]:
    """Convert the renderer's tagged base64 image into a native image observation."""
    output = stdout.decode("utf-8", errors="replace")
    error = stderr.decode("utf-8", errors="replace")
    pattern = re.compile(agent_settings()["model"]["multimodal_regex"])
    images = []
    for match in pattern.finditer(output):
        encoded = match.group(2).split(",", 1)[1]
        if len(encoded) > 4 * ((MAX_PNG + 2) // 3) + 32:
            raise ValueError("Preview PNG exceeds 8 MiB")
        png = base64.b64decode(encoded, validate=True)
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) > MAX_PNG:
            raise ValueError("Invalid preview PNG")
        with Image.open(io.BytesIO(png)) as image:
            if (image.format != "PNG" or not 1 <= image.width <= 4096
                    or not 1 <= image.height <= 4096):
                raise ValueError("Preview must be PNG")
            rgb = image.convert("RGB")
        # Match the native Qwen bridge's bounded image preprocessing.
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        processor = encoder().image_processor
        height, width = smart_resize(
            rgb.height, rgb.width,
            factor=processor.patch_size * processor.merge_size,
            min_pixels=IMAGE_SIZE["shortest_edge"],
            max_pixels=IMAGE_SIZE["longest_edge"])
        images.append({"type": "image", "image": rgb.resize((width, height))})
    feedback = None
    for line in output.splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool) and parsed.get("sourceHash"):
            feedback = parsed
            break
    message = f"Return code: {returncode}\n{pattern.sub('', output)}"
    if error:
        message += "\n" + error
    return ([{"type": "text", "text": message}, *images] if images else message), feedback


class DockerSession:
    """One ephemeral sandbox per trajectory, never used to grade final code."""

    def __init__(self, spec: dict, directory: Path, judge):
        self.spec, self.directory, self.judge = spec, directory, judge
        self.container: str | None = None
        self.commands: list[str] = []
        self.previews: list[dict] = []
        self._owns_slot = False
        self.submitted = False

    async def start(self):
        if self.container:
            return
        if _session_limit is None:
            raise RuntimeError("Configure rollout concurrency before starting an episode")
        await _session_limit.acquire()
        self._owns_slot = True
        name = "artsy-rl-" + uuid4().hex
        try:
            result = await _docker(["run", "-d", "--name", name, "--init",
                                    "--network", "none", "--memory", "2g", "--cpus", "2",
                                    "--workdir", "/workspace", IMAGE, "tail", "-f", "/dev/null"],
                                   timeout=40)
            if result.returncode:
                raise RuntimeError("Could not start Docker episode: " +
                                   result.stderr.decode(errors="replace")[-2048:])
            self.container = name
        except BaseException:
            _session_limit.release()
            self._owns_slot = False
            raise

    async def execute(self, command: str):
        if self.submitted:
            return "Submission already requested.", False
        await self.start()
        self.commands.append(command)
        try:
            result = await _docker(
                ["exec", self.container, "timeout", "--signal=KILL",
                 str(agent_settings()["environment"]["timeout"]), "bash", "-c", command],
                timeout=agent_settings()["environment"]["timeout"] + 2)
        except subprocess.TimeoutExpired:
            return "Return code: 124\nCommand timed out.", False
        if command.strip() == SUBMIT and result.returncode == 0:
            self.submitted = True
        observation, preview = _preview_observation(result.stdout, result.stderr, result.returncode)
        if preview:
            self.previews.append(preview)
        return observation, self.submitted

    async def _read_source(self) -> str:
        if not self.container:
            return ""
        result = await _docker(
            ["exec", self.container, "python", "-c",
             "from pathlib import Path; import sys; "
             "p=Path('/workspace/program.py'); "
             "sys.stdout.buffer.write(p.open('rb').read(65537) if p.is_file() else b'')"],
            timeout=10)
        if result.returncode:
            raise RuntimeError("Cannot read submitted program from sandbox")
        return result.stdout.decode("utf-8") if result.stdout else ""

    async def grade(self, _messages: list[dict]) -> tuple[float, dict[str, float]]:
        """Fresh trusted render, then the same correctness judge as evaluation."""
        self.directory.mkdir(parents=True, exist_ok=True)
        source = await self._read_source()
        result = {"prompt": self.spec["prompt"], "seed": self.spec["seed"],
                  "outcome": "incomplete", "agent": "artsy-cookbook",
                  "commands": self.commands, "previewAttempts": len(self.previews),
                  "scores": {"render_valid": 0.0, "adherence": 0.0}}
        if source.strip() and len(source.encode()) <= SOURCE_BYTES:
            (self.directory / "program.py").write_text(source)
            scored = await _docker(
                ["run", "--rm", "-i", "--network", "none", "--memory", "2g",
                 "--cpus", "2", IMAGE, "python", "/opt/artsy/score_rl.py",
                 "--seed", str(self.spec["seed"])],
                stdin=source.encode(), timeout=90)
            if scored.returncode:
                raise RuntimeError("Trusted Docker scorer failed: " +
                                   scored.stderr.decode(errors="replace")[-2048:])
            response = json.loads(scored.stdout)
            result["feedback"] = response["feedback"]
            if response["feedback"]["ok"]:
                png = base64.b64decode(response["image"], validate=True)
                if len(png) > MAX_PNG:
                    raise ValueError("Scoring PNG exceeds 8 MiB")
                with Image.open(io.BytesIO(png)) as image:
                    if image.format != "PNG" or image.size != (1920, 1200):
                        raise ValueError("Trusted scorer returned an invalid PNG")
                    image.verify()
                (self.directory / "image.png").write_bytes(png)
                result["outcome"] = "submitted"
                result["sourceHash"] = hashlib.sha256(source.encode()).hexdigest()
                result["imageHash"] = hashlib.sha256(png).hexdigest()
                result["scores"]["render_valid"] = 1.0
                response = await self.judge.complete([{"role": "user", "content": [
                    {"type": "text", "text": correctness_judge_prompt(self.spec["prompt"])},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}],
                    getattr(self.judge, "max_tokens", JUDGE_MAX_TOKENS), temperature=0)
                if response.get("finishReason") == "length":
                    raise ValueError("Correctness judge reply truncated")
                judgment = parse_correctness_judgment(response["text"])
                result["judgment"] = judgment
                result["scores"]["adherence"] = judgment["score"]
        (self.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result["scores"]["adherence"], {
            "render_valid": result["scores"]["render_valid"],
            "preview_attempts": len(self.previews),
            "bash_calls": len(self.commands),
        }

    async def close(self):
        try:
            if self.container:
                await _docker(["rm", "-f", self.container], timeout=20)
                self.container = None
        finally:
            if self._owns_slot:
                assert _session_limit is not None
                _session_limit.release()
                self._owns_slot = False


class BashTool:
    """Cookbook Tool protocol with the identical Inspect bash call schema."""

    name = "bash"
    description = "Execute a bash command in the drawing workspace."
    parameters_schema = {"type": "object", "properties": {
        "command": {"type": "string", "description": "The bash command to execute."}},
        "required": ["command"]}

    def __init__(self, session: DockerSession):
        self.session = session

    def to_spec(self):
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters_schema}

    async def run(self, input):
        from tinker_cookbook.tool_use.tools import error_tool_result, simple_tool_result
        command = input.arguments.get("command")
        if not isinstance(command, str):
            return error_tool_result("bash requires a string command",
                                     call_id=input.call_id or "", name=self.name)
        observation, done = await self.session.execute(command)
        return simple_tool_result(observation, call_id=input.call_id or "",
                                  name=self.name, should_stop=done)


class ArtsyGroupBuilder:
    """Tinker Cookbook EnvGroupBuilder protocol, one prompt per rollout group."""

    def __init__(self, spec: dict, group_size: int, step: int, root: Path, judge, max_tokens: int,
                 max_length: int, normalize_rewards: bool):
        self.spec, self.group_size, self.step = spec, group_size, step
        self.root, self.judge = root, judge
        self.max_tokens, self.max_length = max_tokens, max_length
        self.normalize_rewards = normalize_rewards
        self.sessions: list[DockerSession] = []

    async def make_envs(self):
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
        from tinker_cookbook.rl.rollout_limits import RolloutLimits
        from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

        class DrawingMessageEnv(AgentToolMessageEnv):
            async def step(self, message):
                # Match Inspect: a response without a bash call gets a reminder.
                if (not message.get("tool_calls") and not message.get("unparsed_tool_calls")
                        and self._turn_count + 1 < self.max_turns):
                    self._turn_count += 1
                    self.history.extend([message, {"role": "user", "content": NO_TOOL_REMINDER}])
                    return MessageStepResult(0.0, False, self.history)
                return await super().step(message)

        environments = []
        for _ in range(self.group_size):
            session = DockerSession(self.spec,
                self.root / f"step-{self.step:06d}" / self.spec["id"] / uuid4().hex,
                self.judge)
            self.sessions.append(session)

            tool_obj = BashTool(session)
            messages = initial_messages(self.spec, [tool_obj.to_spec()])
            agent = DrawingMessageEnv(tools=[tool_obj], initial_messages=messages,
                                      max_turns=agent_settings()["agent"]["step_limit"],
                                      reward_fn=session.grade,
                                      tool_execution="sequential",
                                      failed_parse_reward=0.0,
                                      terminate_on_parse_error=True)
            environments.append(EnvFromMessageEnv(
                encoder().renderer, agent, failed_parse_reward=0.0,
                terminate_on_parse_error=True, terminate_on_length=True,
                max_trajectory_tokens=self.max_length,
                # Runner clips each response to remaining context space.
                max_generation_tokens=None,
                context_overflow_reward=0.0,
                rollout_limits=RolloutLimits(
                    max_turns=agent_settings()["agent"]["step_limit"],
                    max_trajectory_tokens=self.max_length,
                    max_turn_tokens=self.max_tokens)))
        return environments

    async def cleanup(self):
        await asyncio.gather(*(session.close() for session in self.sessions))
        self.sessions.clear()

    async def compute_group_rewards(self, trajectories, envs):
        """Optional within-prompt variance scaling before Cookbook centers rewards."""
        if not self.normalize_rewards:
            return [(0.0, {}) for _ in trajectories]
        import statistics
        rewards = [sum(turn.reward for turn in traj.transitions) for traj in trajectories]
        scale = statistics.pstdev(rewards) + 1e-8
        center = statistics.mean(rewards)
        # The trainer centers adjusted totals again; this gives normalized advantages.
        return [((reward - center) / scale - reward, {}) for reward in rewards]

    def logging_tags(self):
        return ["artsy", self.spec.get("difficulty", "unspecified")]
