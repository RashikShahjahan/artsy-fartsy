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
import logging
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import threading
from uuid import uuid4

from PIL import Image

from drawing.config import MODEL_ID, MODEL_CONTEXT_WINDOW
from drawing.model_profiles import resolve_model
from drawing.vision import prepare_image
from drawing.agent_contract import drawing_system_prompt, host_user_prompt, turn_countdown
from drawing.judges import (JUDGE_MAX_TOKENS, CODE_SAFETY_MAX_TOKENS, adherence_rubric_prompt,
    parse_adherence_rubric, code_safety_judge_prompt, parse_code_safety_judgment,
    safety_judge_prompt, parse_safety_judgment, RL_JUDGE_VERSION)
from training.data import ROOT
from training.sft import RENDERER, create_encoder

IMAGE = "artsy-rl-runtime"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_PNG = 8 * 1024 * 1024
SOURCE_BYTES = 64 * 1024
NO_TOOL_REMINDER = 'Every response must contain a bash tool call with {"command":"..."}.'

# Leave room for a base64-encoded maximum-sized preview and scorer response.
MAX_OUTPUT = 12 * 1024 * 1024


class RolloutResources:
    """Own capacity for whole groups and their long-lived workspaces."""

    def __init__(self, max_samples: int, group_size: int):
        if max_samples < 1 or group_size < 1 or max_samples < group_size:
            raise ValueError("max_samples must be at least the positive group_size")
        self.group_size = group_size
        self.sessions = asyncio.Semaphore(max_samples)
        self.groups = asyncio.Semaphore(max_samples // group_size)


_resources: RolloutResources | None = None


def configure_concurrency(max_samples: int, group_size: int = 4) -> None:
    """Limit concurrent long-lived Docker workspaces across all prompt groups."""
    global _resources
    _resources = RolloutResources(max_samples, group_size)


def _run(argv: list[str], *, stdin: bytes | None = None, timeout: float = 30) -> subprocess.CompletedProcess:
    """Drain both pipes continuously while retaining only bounded output."""
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outputs = [bytearray(), bytearray()]
    def drain(pipe, output):
        while chunk := pipe.read(65536):
            remaining = MAX_OUTPUT - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
        pipe.close()
    readers = [threading.Thread(target=drain, args=(pipe, output), daemon=True)
               for pipe, output in zip((process.stdout, process.stderr), outputs)]
    for reader in readers:
        reader.start()
    def write_input():
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    writer = None
    if stdin is not None:
        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()
        if writer:
            writer.join()
    return subprocess.CompletedProcess(argv, process.returncode, bytes(outputs[0]), bytes(outputs[1]))


async def _docker(argv: list[str], *, stdin: bytes | None = None, timeout: float = 30):
    # A cancelled to_thread keeps running. Wait for it before allowing its owner
    # to remove the named container or release its capacity.
    task = asyncio.create_task(asyncio.to_thread(_run, ["docker", *argv], stdin=stdin, timeout=timeout))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()  # consume any worker error
        raise


@lru_cache(maxsize=4)
def encoder(model=MODEL_ID, renderer=RENDERER, max_length=MODEL_CONTEXT_WINDOW,
            revision=None):
    # Cached per process; weights live on Tinker, not on this machine.
    return create_encoder(model, max_length, renderer, revision)


@lru_cache(maxsize=1)
def agent_settings():
    import yaml
    return yaml.safe_load((ROOT / "evaluation/mini.yaml").read_text())


def initial_messages(spec: dict, tool_specs: list[dict], model=MODEL_ID,
                     renderer=RENDERER, max_length=MODEL_CONTEXT_WINDOW,
                     revision=None) -> list[dict]:
    """Use the same system/instance framing and API documentation as Inspect."""
    settings = agent_settings()
    prefix = encoder(model, renderer, max_length, revision).renderer.create_conversation_prefix_with_tools(
        tool_specs, system_prompt=settings["agent"]["system_template"])
    prompt = (f'{spec["prompt"]}\n\nPreview with: '
              f'python /opt/artsy/render.py --seed {spec["seed"]} '
              '--output /workspace/preview --preview < /workspace/program.py')
    max_turns = settings["agent"]["step_limit"]
    return [*prefix, {"role": "user", "content": host_user_prompt(
        drawing_system_prompt(token_limit=None), prompt, max_turns)}]


def _preview_observation(stdout: bytes, stderr: bytes, returncode: int, model=MODEL_ID,
                         renderer=RENDERER, max_length=MODEL_CONTEXT_WINDOW,
                         revision=None) -> tuple[list[dict] | str, dict | None]:
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
        processor = encoder(model, renderer, max_length, revision).image_processor
        images.append({"type": "image", "image": prepare_image(
            rgb, processor, vision=resolve_model(model)["vision"])})
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

    def __init__(self, spec: dict, directory: Path, judge, model=MODEL_ID,
                 renderer=RENDERER, max_length=MODEL_CONTEXT_WINDOW, revision=None):
        self.spec, self.directory, self.judge = spec, directory, judge
        self.model, self.renderer = model, renderer
        self.max_length, self.revision = max_length, revision
        self.container: str | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self.commands: list[str] = []
        self.previews: list[dict] = []
        self._owns_slot = False
        self._resources: RolloutResources | None = None
        self.submitted = False
        self.result = None
        self.png = None

    async def start(self):
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Docker episode already closed")
            await self._start()

    async def _start(self):
        if self.container:
            return
        resources = self._resources or _resources
        if resources is None:
            raise RuntimeError("Configure rollout concurrency before starting an episode")
        await resources.sessions.acquire()
        self._resources = resources
        self._owns_slot = True
        name = "artsy-rl-" + uuid4().hex
        self.container = name
        try:
            result = await _docker(["run", "-d", "--name", name, "--init",
                                    "--network", "none", "--memory", "2g", "--cpus", "2",
                                    "--workdir", "/workspace", IMAGE, "tail", "-f", "/dev/null"],
                                   timeout=40)
            if result.returncode:
                raise RuntimeError("Could not start Docker episode: " +
                                   result.stderr.decode(errors="replace")[-2048:])
        except BaseException:
            await self._close()
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
        observation, preview = _preview_observation(
            result.stdout, result.stderr, result.returncode, self.model, self.renderer,
            self.max_length, self.revision)
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

    async def _read_refusal(self):
        if not self.container:
            return None
        response = await _docker(["exec", self.container, "python", "-c",
            "from pathlib import Path; import sys; p=Path('/workspace/refusal.json'); "
            "sys.stdout.buffer.write(p.open('rb').read(4097) if p.is_file() else b'')"], timeout=10)
        if response.returncode:
            raise RuntimeError("Cannot read refusal from sandbox")
        if not response.stdout:
            return None
        try:
            value = json.loads(response.stdout)
            reason = value.get("reason") if isinstance(value, dict) else None
            return reason if isinstance(reason, str) and 0 < len(reason.encode()) <= 2048 else ""
        except (ValueError, UnicodeDecodeError):
            return ""

    async def _judge(self, role, prompt, parser, png=None, max_tokens=None):
        client = self.judge[role]
        if max_tokens is None:
            max_tokens = getattr(client, "max_tokens", JUDGE_MAX_TOKENS)
        content = [{"type": "text", "text": prompt}]
        if png is not None:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        response = await client.complete([{"role": "user", "content": content}],
                                         max_tokens, temperature=0)
        if response.get("finishReason") == "length":
            raise ValueError(role + " judge reply truncated")
        return parser(response["text"])

    async def grade(self, _messages: list[dict]) -> tuple[float, dict[str, float]]:
        """Collect independent safety/adherence signals; group reward is computed later."""
        self.directory.mkdir(parents=True, exist_ok=True)
        source = await self._read_source()
        refusal = await self._read_refusal()
        result = {"prompt": self.spec["prompt"], "seed": self.spec["seed"],
                  "outcome": "incomplete", "agent": "artsy-cookbook",
                  "commands": self.commands, "previewAttempts": len(self.previews),
                  "scores": {"render_valid": 0.0, "adherence": 0.0},
                  "judgeVersion": RL_JUDGE_VERSION, "unsafe": False,
                  "refused": refusal is not None, "refusalCorrect": False,
                  "expectedRefusal": self.spec.get("kind") == "refusal"}
        judgment = await self._judge("code_safety",
            code_safety_judge_prompt(self.spec["prompt"], source, self.commands),
            parse_code_safety_judgment, max_tokens=CODE_SAFETY_MAX_TOKENS)
        result["codeSafety"] = judgment
        result["unsafe"] = judgment["malicious"]
        if refusal is not None:
            result["refusalCorrect"] = bool(refusal and not source and self.submitted
                                             and result["expectedRefusal"])
            result["outcome"] = "refused" if refusal and not source else "invalid_refusal"
        if (not result["unsafe"] and refusal is None and source.strip()
                and len(source.encode()) <= SOURCE_BYTES):
            (self.directory / "program.py").write_text(source)
            scorer_name = "artsy-score-" + uuid4().hex
            try:
                scored = await _docker(
                    ["run", "--rm", "-i", "--name", scorer_name, "--network", "none", "--memory", "2g",
                     "--cpus", "2", IMAGE, "python", "/opt/artsy/score_rl.py",
                     "--seed", str(self.spec["seed"])],
                    stdin=source.encode(), timeout=90)
            finally:
                try:
                    removal = await _docker(["rm", "-f", scorer_name], timeout=20)
                    if removal.returncode and b"No such container" not in removal.stderr:
                        logging.error("Failed to remove scorer container %s: %s", scorer_name,
                                      removal.stderr.decode(errors="replace")[-2048:])
                except Exception as error:
                    logging.error("Failed to remove scorer container %s: %s", scorer_name, error)
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
                self.png = png
                safety, adherence = await asyncio.gather(
                    self._judge("safety", safety_judge_prompt(self.spec["prompt"]),
                                parse_safety_judgment, png),
                    self._judge("adherence", adherence_rubric_prompt(self.spec["prompt"]),
                                parse_adherence_rubric, png))
                result["imageSafety"] = safety
                result["unsafe"] |= safety["harmful"]
                result["judgment"] = adherence
                result["scores"]["adherence"] = adherence["score"] / 4.0
        self.result = result
        (self.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return 0.0, {
            "render_valid": result["scores"]["render_valid"],
            "preview_attempts": len(self.previews),
            "bash_calls": len(self.commands),
        }

    async def close(self):
        async with self._lifecycle_lock:
            self._closed = True
            await self._close()

    async def _close(self):
        try:
            if self.container:
                result = await _docker(["rm", "-f", self.container], timeout=20)
                if result.returncode and b"No such container" not in result.stderr:
                    raise RuntimeError("Could not remove Docker episode: " + result.stderr.decode(errors="replace")[-2048:])
                self.container = None
        finally:
            if self._owns_slot:
                assert self._resources is not None
                self._resources.sessions.release()
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
                 max_length: int, normalize_rewards: bool, model=MODEL_ID, renderer=RENDERER,
                 revision=None):
        self.spec, self.group_size, self.step = spec, group_size, step
        self.root, self.judge = root, judge
        self.max_tokens, self.max_length = max_tokens, max_length
        self.normalize_rewards = normalize_rewards
        self.model, self.renderer, self.revision = model, renderer, revision
        self.sessions: list[DockerSession] = []
        self._owns_group_slot = False
        self._resources: RolloutResources | None = None

    async def make_envs(self):
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageStepResult
        from tinker_cookbook.rl.rollout_limits import RolloutLimits
        from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

        class DrawingMessageEnv(AgentToolMessageEnv):
            def _countdown(self):
                turn = self._turn_count + 1
                self.history.append({"role": "user", "content":
                    turn_countdown(turn, self.max_turns)})

            async def initial_observation(self):
                if not self.history:
                    await super().initial_observation()
                    self._countdown()
                return self.history

            async def step(self, message):
                # Match Inspect: a response without a bash call gets a reminder.
                if (not message.get("tool_calls") and not message.get("unparsed_tool_calls")
                        and self._turn_count + 1 < self.max_turns):
                    self._turn_count += 1
                    self.history.extend([message, {"role": "user", "content": NO_TOOL_REMINDER}])
                    result = MessageStepResult(0.0, False, self.history)
                else:
                    result = await super().step(message)
                if not result.episode_done:
                    self._countdown()
                return result

        # Admit whole groups so partially started groups cannot exhaust slots
        # while waiting for trajectories needed before group cleanup.
        resources = _resources
        if resources is None:
            raise RuntimeError("Configure rollout concurrency before creating a group")
        if self.group_size != resources.group_size:
            raise ValueError("Group size does not match configured concurrency")
        await resources.groups.acquire()
        self._resources = resources
        self._owns_group_slot = True
        try:
            return self._make_environments(DrawingMessageEnv, EnvFromMessageEnv, RolloutLimits)
        except BaseException:
            await self.cleanup()
            raise

    def _make_environments(self, DrawingMessageEnv, EnvFromMessageEnv, RolloutLimits):
        environments = []
        for _ in range(self.group_size):
            session = DockerSession(self.spec,
                self.root / f"step-{self.step:06d}" / self.spec["id"] / uuid4().hex,
                self.judge, self.model, self.renderer, self.max_length, self.revision)
            session._resources = self._resources
            self.sessions.append(session)

            tool_obj = BashTool(session)
            messages = initial_messages(self.spec, [tool_obj.to_spec()], self.model,
                                        self.renderer, self.max_length, self.revision)
            agent = DrawingMessageEnv(tools=[tool_obj], initial_messages=messages,
                                      max_turns=agent_settings()["agent"]["step_limit"],
                                      reward_fn=session.grade,
                                      tool_execution="sequential",
                                      failed_parse_reward=0.0,
                                      terminate_on_parse_error=True)
            environments.append(EnvFromMessageEnv(
                encoder(self.model, self.renderer, self.max_length, self.revision).renderer,
                agent, failed_parse_reward=0.0,
                terminate_on_parse_error=True, terminate_on_length=True,
                max_trajectory_tokens=self.max_length,
                # Runner clips each response to remaining context space.
                # Reserve one output token: an exactly full prompt cannot sample.
                # The runner still clips the actual response to remaining capacity.
                max_generation_tokens=1,
                context_overflow_reward=0.0,
                rollout_limits=RolloutLimits(
                    max_turns=agent_settings()["agent"]["step_limit"],
                    max_trajectory_tokens=self.max_length,
                    max_turn_tokens=self.max_tokens)))
        return environments

    async def cleanup(self):
        try:
            results = await asyncio.gather(*(session.close() for session in self.sessions), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logging.error("Failed to clean up Docker episode: %s", result)
        finally:
            self.sessions.clear()
            if self._owns_group_slot:
                assert self._resources is not None
                self._resources.groups.release()
                self._owns_group_slot = False

    async def compute_group_rewards(self, trajectories, envs):
        """Return final reward adjustments, leaving mean-centering to Cookbook."""
        if len(trajectories) != len(self.sessions):
            raise ValueError("Trajectory/session count mismatch")
        # Tool/turn/length limit terminations may bypass the terminal reward hook.
        for session in self.sessions:
            if session.result is None:
                await session.grade([])
        adjusted = []
        for session, trajectory in zip(self.sessions, trajectories):
            result = session.result
            if result["unsafe"]:
                reward = -1.0
            elif result.get("refusalCorrect"):
                reward = 1.0
            elif (result.get("refused") or self.spec.get("kind") == "refusal"
                  or not session.submitted or not result["scores"]["render_valid"]):
                reward = 0.0
            else:
                reward = result["scores"]["adherence"]
            result["scores"].update(reward=reward)
            session.directory.mkdir(parents=True, exist_ok=True)
            (session.directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            terminal = sum(turn.reward for turn in trajectory.transitions)
            adjusted.append((reward - terminal, {"reward": reward}))
        return adjusted

    def logging_tags(self):
        return ["artsy", self.spec.get("difficulty", "unspecified")]
