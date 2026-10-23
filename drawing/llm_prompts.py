"""LLM-authored drawing prompts, balanced by an independent skill taxonomy."""
from __future__ import annotations

import difflib
import re
from typing import Any

from .dataset_client import parse_json_reply
from .dataset_prompts import DIFFICULTIES, FAMILIES, SKILL_TAXONOMY, compact_json, sha256, validate_specs

GENERATOR_VERSION = "artsy-llm-prompts-v3"


def _slots(count: int, seed: int, mode: str = "standard", difficulty: str = "mixed") -> list[dict[str, Any]]:
    if type(count) is not int or not 1 <= count <= 10_000:
        raise ValueError("count must be an integer from 1 to 10000")
    if type(seed) is not int or not 0 <= seed <= 9_007_199_254_740_991:
        raise ValueError("seed must be a nonnegative safe integer")
    if mode not in ("standard", "repair"):
        raise ValueError("mode must be standard or repair")
    if difficulty not in (*DIFFICULTIES, "mixed"):
        raise ValueError(f"difficulty must be mixed or one of {DIFFICULTIES}")
    if mode == "repair" and difficulty != "mixed":
        raise ValueError("The legacy repair curriculum does not accept difficulty selection")
    families = [family for family, _ in FAMILIES]
    skills = list(SKILL_TAXONOMY)
    slots = []
    for index in range(count):
        level = ("very-hard" if mode == "repair" else
                 DIFFICULTIES[index % len(DIFFICULTIES)] if difficulty == "mixed" else difficulty)
        slots.append({
            "index": index,
            "family": families[(index + seed) % len(families)],
            "primarySkill": skills[(index + seed) % len(skills)],
            "difficulty": level,
        })
    return slots


def _words(prompt: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", prompt.lower()) if len(word) >= 3}


def _too_similar(prompt: str, earlier: list[str]) -> bool:
    normalized = re.sub(r"\s+", " ", prompt.strip()).lower()
    words = _words(normalized)
    for previous in earlier:
        other = _words(previous)
        overlap = len(words & other) / len(words | other) if words | other else 1.0
        if normalized == previous or overlap >= 0.82:
            return True
        if overlap >= 0.55 and difflib.SequenceMatcher(None, normalized, previous).ratio() >= 0.88:
            return True
    return False


def _parse_batch(text: str, slots: list[dict[str, Any]], earlier: list[str], seed: int,
                 *, mode: str) -> list[dict[str, Any]]:
    reply = parse_json_reply(text)
    if not isinstance(reply, dict) or set(reply) != {"items"} or not isinstance(reply["items"], list):
        raise ValueError("Prompt generator must return JSON with an items array")
    if len(reply["items"]) != len(slots):
        raise ValueError("Prompt generator returned the wrong number of items")
    specs = []
    prompts = list(earlier)
    for slot, item in zip(slots, reply["items"]):
        if not isinstance(item, dict) or type(item.get("index")) is not int or item["index"] != slot["index"]:
            raise ValueError("Prompt generator returned a missing or out-of-order slot index")
        required = {"index", "prompt", "skills"} | ({"technicalSkills"} if mode == "repair" else set())
        if set(item) != required:
            raise ValueError("Unexpected prompt generator fields")
        prompt = item["prompt"]
        if (not isinstance(prompt, str) or len(prompt.strip()) < 45 or len(prompt) > 1500
                or len(prompt.split()) < 9 or any(ord(char) < 32 and char not in "\n\t" for char in prompt)
                or "```" in prompt):
            raise ValueError("Generated prompt must be a substantive plain-language drawing request")
        skills = item["skills"]
        if (not isinstance(skills, list) or not 1 <= len(skills) <= 5
                or any(not isinstance(skill, str) or skill not in SKILL_TAXONOMY for skill in skills)
                or len(set(skills)) != len(skills) or slot["primarySkill"] not in skills):
            raise ValueError("Generated skill labels must be unique taxonomy members including the primary skill")
        normalized = re.sub(r"\s+", " ", prompt.strip()).lower()
        if _too_similar(normalized, prompts):
            raise ValueError("Generated prompt is a duplicate or near-duplicate")
        prompts.append(normalized)
        index = slot["index"]
        entropy = sha256(f"{GENERATOR_VERSION}:{seed}:{index}")
        spec: dict[str, Any] = {
            "id": entropy[:24], "family": slot["family"], "prompt": prompt.strip(),
            "seed": int(entropy[24:32], 16), "skills": skills, "difficulty": slot["difficulty"],
        }
        if mode == "repair":
            technical = item["technicalSkills"]
            if (not isinstance(technical, list) or not 2 <= len(technical) <= 6
                    or any(not isinstance(x, str) or not x.strip() or len(x) > 80 for x in technical)
                    or len(set(technical)) != len(technical)):
                raise ValueError("Repair tasks need two to six distinct technical skills")
            spec["repairTask"] = {"complexity": "advanced", "skills": technical}
        specs.append(spec)
    return validate_specs(specs)


async def generate_llm_prompts(client: Any, count: int = 20, seed: int = 42, *,
                               mode: str = "standard", difficulty: str = "mixed",
                               batch_size: int = 5,
                               max_tokens: int = 4096, retries: int = 3) -> dict[str, Any]:
    """Produce a validated prompt plan, preserving raw response provenance for auditing.

    Slot labels and seeds are deterministic; model-authored wording is frozen by
    the CLI's saved plan and may not be reproducible from the same seed alone.
    """
    slots = _slots(count, seed, mode, difficulty)
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("prompt batch size must be between 1 and 8")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("prompt max tokens must be positive")
    if type(retries) is not int or not 1 <= retries <= 5:
        raise ValueError("prompt retries must be between 1 and 5")
    system = (
        "You create diverse, self-contained drawing requests for evaluating and training a "
        "Python/ArtCanvas image-drawing agent on a 1920x1200 canvas. Write ORIGINAL prompts, "
        "not permutations of stock templates. Make all tasks benign and feasible with basic "
        "geometry, math, random, shapes, fills, paths, and gradients; no external images, "
        "files, fonts, network access, installed packages, or user-provided code. "
        "Each prompt is 1-5 natural sentences with concrete visual success criteria, "
        "but never include a hidden answer, a scoring rubric, source code, or instructions "
        "to the evaluator. Balance creative scenes with measurable geometry and constraints. "
        "Follow each slot's difficulty: easy means one or two shapes and few constraints; "
        "medium means several objects with layout or layering; hard means complex geometry "
        "and multiple interacting constraints; very-hard means challenging procedural "
        "geometry or precise interdependent conditions. Keep every level solvable. "
        "Difficulty describes task complexity, not a required failure or repair. "
        "Never deliberately introduce defects or instruct the agent to repair code. "
        "For legacy mode=repair, also name 2-6 genuine computational skills as technicalSkills. "
        "Return ONLY one JSON object with an items array in slot order. Each item has "
        "index (integer), prompt (string), and skills (1-5 taxonomy labels, including "
        "primarySkill); include technicalSkills (array of strings) ONLY in mode=repair."
    )
    specs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for offset in range(0, count, batch_size):
        batch_slots = slots[offset:offset + batch_size]
        recent = [s["prompt"] for s in specs[-30:]]
        request = {
            "version": GENERATOR_VERSION, "mode": mode,
            "taxonomy": SKILL_TAXONOMY, "slots": batch_slots,
            "recentPromptsToAvoid": recent,
        }
        failure = "unknown error"
        for attempt in range(retries):
            response = await client.complete([
                {"role": "system", "content": system},
                {"role": "user", "content": compact_json(request)},
            ], max_tokens)
            try:
                if response.get("finishReason") == "length":
                    raise ValueError("Prompt generator reply was truncated")
                generated = _parse_batch(response["text"], batch_slots,
                                         [s["prompt"] for s in specs], seed, mode=mode)
            except (ValueError, KeyError, TypeError) as error:
                failure = str(error)
                request["previousError"] = failure
                continue
            specs.extend(generated)
            batches.append({"indices": [slot["index"] for slot in batch_slots],
                            "attempts": attempt + 1, "completion": response})
            break
        else:
            raise ValueError(f"Could not generate valid prompts for slots {offset}-{offset + len(batch_slots) - 1}: {failure}")
    return {"specs": validate_specs(specs), "batches": batches}
