# Running Artsy visual RLHF

This path is independent of SFT. It fine-tunes a Qwen3.5-9B multimodal preference judge with human A/B/Tie labels, freezes that Tinker sampler checkpoint, then uses the trained judge to provide relative visual rewards for Artsy's existing Qwen3.5-4B multi-turn agent.

## Prepare human image comparisons

Store human-labeled pairs in a JSONL file, with image paths relative to the JSONL file:

\`\`\`json
{"pair_id":"pair-0001","group_id":"cabin-family","image_a":"images/cabin-a.png","image_b":"images/cabin-b.png","winner":"B","split":"train"}
{"pair_id":"pair-0002","group_id":"skyline-family","image_a":"images/skyline-a.png","image_b":"images/skyline-b.png","winner":"A","split":"validation"}
\`\`\`

\`winner\` must be \`A\`, \`B\`, or \`Tie\`. Split values are \`train\`, \`validation\`, and \`test\`. Keep entire prompt families in one split, and keep validation/test labels out of training. The validator also rejects duplicate pairs, matching images and image paths outside the dataset directory.

Validate without a Tinker training call:

\`\`\`bash
uv run python -m preference.train --data datasets/preference-v1/labels.jsonl --validate-only
\`\`\`

## Train the visual judge

\`\`\`bash
uv run python -m preference.train \
  --data datasets/preference-v1/labels.jsonl \
  --model Qwen/Qwen3.5-9B \
  --epochs 1 --batch-size 4 --rank 16
\`\`\`

The script supervises only the assistant's preference label and augments pairs with reversed image order. Save the emitted \`tinker://...\` **sampler checkpoint** separately from the policy's SFT weights. This is a generative A/B/Tie preference classifier, not an unsupported scalar reward head.

Evaluate agreement and A/B-order consistency on held-out human labels:

\`\`\`bash
uv run python -m preference.evaluate \
  --data datasets/preference-v1/labels.jsonl \
  --split validation \
  --sampler-path tinker://YOUR_VISUAL_JUDGE_SAMPLER
\`\`\`

To measure the off-the-shelf model, omit \`--sampler-path\`.

## Configure the RLHF judges

Configure these settings in the local \`.env\` (never commit credentials):

- \`ARTSY_CORRECTNESS_JUDGE_MODEL\`, \`ARTSY_CORRECTNESS_JUDGE_URL\`: existing Tinker/OpenAI-compatible rubric judge for instruction adherence.
- \`ARTSY_SAFETY_JUDGE_MODEL\`, \`ARTSY_SAFETY_JUDGE_URL\`: independent content safety judge.
- \`ARTSY_CODE_SAFETY_JUDGE_MODEL\`, \`ARTSY_CODE_SAFETY_JUDGE_URL\`: independent full-code/action-history safety judge.
- Set corresponding \`*_API_KEY\` variables if the particular provider requires authentication.
- \`ARTSY_VISUAL_JUDGE_MODEL=Qwen/Qwen3.5-9B\`
- \`ARTSY_VISUAL_JUDGE_SAMPLER_PATH=tinker://YOUR_VISUAL_JUDGE_SAMPLER\`

The visual judge uses Tinker's native image sampler; it does not need the Artsy inference server. For Tinker-native rubric/safety judges set each \`*_URL=tinker://native\` and use a supported VLM model/renderer.

## Train the policy

Use the *existing SFT checkpoint* as policy initialization and an RL prompt JSONL that includes both benign and trusted \`kind=refusal\` harmful examples:

\`\`\`bash
uv run python -m training.grpo \
  --dataset datasets/grpo-v1/prompts.jsonl \
  --init-checkpoint checkpoints/sft-v1/final.json \
  --output-dir checkpoints/grpo-rlhf-v1 \
  --group-size 4 --max-samples 4 --epochs 1 --max-steps 1 \
  --visual-sampler-path tinker://YOUR_VISUAL_JUDGE_SAMPLER \
  --dry-run
\`\`\`

Remove \`--dry-run\` after verifying the plan and credentials. Artsy uses its trusted Docker renderer and native Cookbook multi-turn rollouts. The group reward ranks final PNGs pairwise (including reversed order), combines visual preference and 0–4 rubric adherence equally, and applies hard safety/refusal outcomes. The Cookbook itself mean-centers group rewards. Avoid the old \`--normalize-rewards\` variance scaling; the switch is retained only for CLI compatibility and has no effect under this reward.

Inspect remains the independent held-out evaluation harness. Do not use the frozen visual judge's self-reported win rate as the sole measure of improvement: also measure human preference win rate, held-out adherence, safety failures, benign over-refusal and render success.

Live Tinker image training/sampling and full Docker smoke tests require configured credentials and a working local container runtime.
