"""Resume a prompt-only plan through OpenRouter's asynchronous Batch API."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .dataset import _write_json, _write_jsonl_atomic
from .dataset_specs import SKILL_TAXONOMY, compact_json, sha256, validate_specs
from .generate_dataset import _load_env_file
from .llm_prompts import (AUTHOR_SYSTEM, GENERATOR_VERSION, THEMES, _parse_batch,
                          _slots, generate_llm_prompts)

API = "https://openrouter.ai/api/v1/batches"
FINAL = {"completed", "failed", "expired", "cancelled"}


def _request(url: str, key: str, payload: dict | None = None) -> dict:
    body = compact_json(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, body, {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        # Provider error bodies can echo request content or credentials.
        raise RuntimeError(f"OpenRouter batch API returned HTTP {error.code}") from None
    if not isinstance(result, dict):
        raise RuntimeError("OpenRouter batch API returned invalid JSON")
    return result


def _completion(result: dict, model: str) -> dict:
    if result.get("error"):
        raise ValueError(str(result["error"].get("message", "Batch request failed"))[:300])
    response = result.get("response") or {}
    if response.get("status_code") != 200:
        raise ValueError(f"Batch request returned HTTP {response.get('status_code')}")
    body = response.get("body") or {}
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Batch request returned no text completion")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    details = message.get("reasoning_details")
    usage = body.get("usage")
    return {
        "text": content,
        **({"reasoning": reasoning} if isinstance(reasoning, str) and reasoning.strip() else {}),
        **({"reasoning_details": details} if isinstance(details, list) and details else {}),
        "model": body.get("model") or model,
        **({"usage": usage} if isinstance(usage, dict) else {}),
        "finishReason": choice.get("finish_reason") or "unknown",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds must be at least 10")
    _load_env_file()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("Set OPENROUTER_API_KEY")
    root = args.output
    progress = json.loads((root / "prompt-progress.json").read_text())
    config = progress["config"]
    if (config["version"] != GENERATOR_VERSION or
            config["authorInstructionsHash"] != sha256(AUTHOR_SYSTEM) or
            config["baseUrl"].rstrip("/") != "https://openrouter.ai/api/v1"):
        parser.error("Prompt run is incompatible with this batch author")
    slots = _slots(config["count"], config["seed"], config["difficulty"],
                   config["harmfulCount"], config["maliciousCodeCount"],
                   config["benignBoundaryCount"])
    lock = root / ".prompt-generation.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        parser.error(f"Prompt generation is already locked: {lock}")
    try:
        records_dir = root / "prompt-records"
        state_path = root / "openrouter-batch-state.json"
        while True:
            accepted = {int(path.stem): json.loads(path.read_text())
                        for path in records_dir.glob("*.json")}
            if len(accepted) == config["count"]:
                break
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not state.get("id"):
                remaining = [slot for slot in slots if slot["index"] not in accepted]
                if any(state.get("attempts", {}).get(str(slot["index"]), 0) >= config["retries"]
                       for slot in remaining):
                    raise RuntimeError("A slot exhausted batch retries; inspect openrouter-batch-state.json")
                # Batch requests share the committed context available at submit time.
                # This avoids changing any already checkpointed prompt.
                recent_indices = sorted(accepted)[-30:]
                quotas = {theme: sum(slot["contentTheme"] == theme for slot in slots)
                          for theme in THEMES}
                requests = []
                contexts = {}
                for slot in remaining:
                    index = slot["index"]
                    related = [i for i in sorted(accepted)
                               if slots[i]["contentTheme"] == slot["contentTheme"]]
                    avoid = list(dict.fromkeys(related + recent_indices))
                    recent = [accepted[i]["spec"]["prompt"] for i in avoid]
                    context = {"version": GENERATOR_VERSION, "taxonomy": SKILL_TAXONOMY,
                               "slots": [slot], "recentPromptsToAvoid": recent,
                               "plannedThemeQuotas": quotas}
                    failure = state.get("failures", {}).get(str(index))
                    if failure:
                        context.update(previousError=failure["reason"],
                                       previousDraft=failure.get("draft"))
                    body = {"messages": [
                        {"role": "system", "content": AUTHOR_SYSTEM},
                        {"role": "user", "content": compact_json(context)},
                    ], "reasoning": {"enabled": True, "exclude": False}}
                    if config["maxTokens"] is not None:
                        body["max_tokens"] = config["maxTokens"]
                    requests.append({"custom_id": f"prompt-{index:06d}", "body": body})
                    contexts[str(index)] = recent
                payload = {"endpoint": "/v1/chat/completions", "model": config["model"],
                           "requests": requests}
                # Persist the pending request before the external side effect.
                state.update(pendingHash=sha256(compact_json(payload)),
                             indices=[slot["index"] for slot in remaining],
                             contexts=contexts)
                _write_json(state_path, state)
                batch = _request(API, key, payload)
                if not isinstance(batch.get("id"), str):
                    raise RuntimeError("OpenRouter did not return a batch ID")
                state["id"] = batch["id"]
                state.pop("pendingHash", None)
                _write_json(state_path, state)
                print(f"Submitted batch {batch['id']} with {len(remaining)} requests", flush=True)
            try:
                batch = _request(f"{API}/{state['id']}", key)
            except RuntimeError as error:
                # Newly accepted batches can briefly return 404 before the read
                # endpoint sees them. Keep the saved ID and retry, never resubmit.
                if str(error).endswith("HTTP 404"):
                    time.sleep(args.poll_seconds)
                    continue
                raise
            counts = batch.get("request_counts") or {}
            print(f"Batch {state['id']}: {batch.get('status')} "
                  f"({counts.get('completed', 0)}/{counts.get('total', '?')} completed)", flush=True)
            if batch.get("status") not in FINAL or (batch.get("status") == "completed"
                                                     and batch.get("results") is None):
                time.sleep(args.poll_seconds)
                continue
            if batch.get("status") != "completed":
                raise RuntimeError(f"Batch ended with status {batch.get('status')}: "
                                   f"{(batch.get('error') or {}).get('message', '')}")
            results = batch["results"]
            by_id = {result["custom_id"]: result for result in results}
            if len(by_id) != len(state["indices"]):
                raise RuntimeError("Batch result count did not match submitted requests")
            for index in state["indices"]:
                slot = slots[index]
                result = by_id[f"prompt-{index:06d}"]
                attempt = state.get("attempts", {}).get(str(index), 0) + 1
                response = None
                try:
                    response = _completion(result, config["model"])
                    if response["finishReason"] == "length":
                        raise ValueError("Prompt generator reply was truncated")
                    prior = [record["spec"]["prompt"] for record in accepted.values()]
                    spec = _parse_batch(response["text"], [slot], prior, config["seed"])[0]
                    record = {"index": index, "spec": spec, "attempts": attempt,
                              "completion": response, "slot": slot,
                              "recentPromptsToAvoid": state["contexts"][str(index)],
                              "batchId": state["id"]}
                    with (root / "prompt-attempts.jsonl").open("a") as stream:
                        stream.write(compact_json({**record, "status": "accepted"}) + "\n")
                    _write_json(records_dir / f"{index:06d}.json", record)
                    accepted[index] = record
                    state.get("failures", {}).pop(str(index), None)
                except (ValueError, KeyError, TypeError) as error:
                    state.setdefault("attempts", {})[str(index)] = attempt
                    state.setdefault("failures", {})[str(index)] = {
                        "reason": str(error)[:300],
                        "draft": response.get("text") if response else None,
                    }
                    with (root / "prompt-attempts.jsonl").open("a") as stream:
                        stream.write(compact_json({"index": index, "attempts": attempt,
                            "slot": slot, "completion": response, "status": "rejected",
                            "reason": str(error)[:300], "batchId": state["id"]}) + "\n")
            state.setdefault("batches", []).append({
                "id": state["id"], "status": batch["status"], "usage": batch.get("usage"),
                "requestCounts": batch.get("request_counts")})
            state.pop("id")
            state.pop("indices", None)
            state.pop("contexts", None)
            _write_json(state_path, state)
        # Reuse the ordinary generator's checkpoint verification and plan audit.
        class NoCalls:
            async def complete(self, *_args, **_kwargs):
                raise AssertionError("All prompt records are already complete")
        generated = asyncio.run(generate_llm_prompts(NoCalls(), config["count"],
            config["seed"], difficulty=config["difficulty"],
            harmful_count=config["harmfulCount"],
            malicious_code_count=config["maliciousCodeCount"],
            benign_boundary_count=config["benignBoundaryCount"],
            batch_size=config["batchSize"], max_tokens=config["maxTokens"],
            retries=config["retries"], completed=accepted))
        specs = validate_specs(generated["specs"])
        _write_json(root / "prompt-plan.json", {"config": config, "specs": specs,
            "specsHash": sha256(compact_json(specs)), "batches": generated["batches"],
            "audit": generated["audit"], "instructions": {"author": AUTHOR_SYSTEM},
            "openrouterBatches": state.get("batches", [])})
        _write_jsonl_atomic(root / "prompts.jsonl", specs)
        print(f"Saved {len(specs)} prompts to {root / 'prompts.jsonl'}", flush=True)
        return 0
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
