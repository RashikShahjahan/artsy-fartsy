---
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: image-text-to-text
language:
- en
tags:
- lora
- artcanvas
- pycairo
- tool-use
- drawing
- sft
---

# Artsy SFT: Supervised drawing and tool use

This is the supervised fine-tuning stage of Artsy, a drawing agent that turns a text request into an executable ArtCanvas Python program. It learns to write and edit code, render a preview, inspect the resulting image, and submit a finished drawing through a mini-SWE tool loop.

This repository contains a **LoRA adapter**, loaded on top of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It generates code and tool calls; the drawing environment executes the program to produce the image.

## Training

Trained on **2,282 trajectories** for **72 updates**, with batches of 32 trajectories, learning rate `2e-4`, and a cosine schedule. The recorded loss policy is `final-write-final-preview-submit`, concentrating supervision on the final write, preview, and submission actions. Training includes multimodal preview observations.

All three stages use LoRA rank **128**, the pinned Qwen3.5-4B base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and Tinker for training. The sequence is:

[SFT](https://huggingface.co/Rashik24/parti-sft-artcanvas-full-2026-11-07-qwen3.5-4b-r128-b32) → [RLVR](https://huggingface.co/Rashik24/rlvr-experiment9-genai681-qwen3.5-4b-r128-b32-g4) → [RLHF](https://huggingface.co/Rashik24/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery)

## Evaluation

In the repository's comparison on **150 hard GenAI-Bench prompts**, using the same mini-SWE harness and a **12-turn cap**, this checkpoint delivered **119/150 valid images** with a mean normalized HPSv3 score of **0.348** across all episodes. The base model delivered **102/150** with a score of **0.094**. Missing images count as zero in the normalized mean.

These are descriptive results from one checkpoint comparison, not a human approval rate or a multi-seed study. See the [evaluation notes and charts](https://github.com/RashikShahjahan/artsy-fartsy/tree/main/docs/model-development) for the protocol and source data.

## Loading the adapter

The exported adapter was checked with Transformers `5.5.4` and PEFT `0.21.3`.

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel

base_id = "Qwen/Qwen3.5-4B"
base_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
processor = AutoProcessor.from_pretrained(base_id, revision=base_revision)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_id, revision=base_revision, dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(
    base, "Rashik24/parti-sft-artcanvas-full-2026-11-07-qwen3.5-4b-r128-b32", revision="ee57fed866a16653de398be0a42f1105520d1572"
).eval()
```

Use `enable_thinking=False` when applying the chat template. To reproduce the drawing behavior, provide the [Artsy instructions, ArtCanvas API, and mini-SWE execution environment](https://github.com/RashikShahjahan/artsy-fartsy#readme), including rendered preview images. Loading the weights alone does not provide those tools.

The adapter includes an output-head update. Keep it attached through PEFT; merging must preserve the base input embeddings separately. Run generated Python in the drawing sandbox. Reliability and image quality depend on the tool protocol, renderer, prompt, and execution budget.
