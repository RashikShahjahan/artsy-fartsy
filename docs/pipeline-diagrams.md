# Datagen, evaluation, and training flows

These diagrams describe the repository's current Python pipeline. They distinguish
dataset-generation review from benchmark evaluation: the former produces candidate
training data, while the latter measures a model on a fixed test set.

## Dataset generation (`drawing.generate_dataset`)

### 1. Prompt planning

```mermaid
flowchart TD
    A[CLI options and .env] --> B{Prompt source?}
    B -- --input --> C[Read parseable JSONL specs]
    B -- Built-in curriculum --> D[Make seeded balanced slots]
    D --> E[Author writes one request per slot]
    E --> F{Parseable prompt and slot identity?}
    F -- No --> R[Retry protocol/API error up to 5 times]
    R --> E
    F -- Yes --> G[Checkpoint raw prompt without content review]
    G --> H[Freeze prompt-plan.json and write prompts.jsonl]
    C --> H
    H --> I{--prompts-only?}
    I -- Yes --> J[Exit]
    I -- No --> K[Generate candidates]
```

The default plan contains 35 ordinary drawings and 5 each of benign boundary,
harmful-image refusal and malicious-code refusal requests. Quotas and authoring
instructions guide generation; no reviewer, wording, label-match, length or
similarity gate filters the results. IDs and required data types remain validated.
Each successful prompt uses only one author call. New plans use generator v11;
use a fresh output directory for the changed configuration.

### 2. Candidate generation and human curation

```mermaid
flowchart TD
    A[Prompt specs] --> B{Authored refusal?}
    B -- Yes --> C[Save refusal target for human review]
    B -- No --> D[Teacher returns code or refusal]
    D --> E{Parseable response?}
    E -- No --> R[Repair within configured budget]
    R --> D
    E -- Refusal --> C
    E -- Code --> F[Render preview under execution limits]
    F -- Execution failure --> R
    F -- Valid PNG --> G[Fresh final render]
    G -- Execution failure --> R
    G -- Success --> H[Save source, PNG, hashes and trace for human review]
    C --> I[Checkpoint every candidate]
    H --> I
    R -- Exhausted --> I
    F -- Infrastructure failure --> I
    G -- Infrastructure failure --> I
    I --> J[Write candidates.jsonl, manifest.json and summary.json]
    J --> K[Human approves or rejects candidates]
    K --> L[Verify recorded artifacts and convert trajectory]
    L --> M[Export manually approved records to accepted.jsonl]
```

Dataset generation makes no prompt-review, code-monitor or image-judge calls,
even when evaluation judge environment variables are configured. It does not
reject duplicates or grade visual quality. New drawing and refusal records use
`review` status; execution failures remain visible with their recorded outcomes.
No automated status, selection audit or historical judge verdict blocks conversion
of a human-selected record. Hashes, paths, source/trace consistency, runtime
compatibility and usable recorded actions are still checked. Existing frozen
datasets are left intact; new candidate generation uses dataset v6.

## Evaluation (`evaluation.evaluate`)

```mermaid
flowchart TD
    A[evaluation/generated-110.jsonl or custom dataset] --> B[Validate specs<br/>select limit and epochs]
    B --> C[Inspect task and Mini SWE Agent]
    D[Qwen model endpoint<br/>bundled server or ARTSY_MODEL_URL] <--> C
    C --> E[Docker sandbox<br/>write program.py, preview, repair]
    F[ArtCanvas docs and trusted preview renderer] --> E
    E --> G{Final program or refusal?}
    G -- Program --> CS{Audit Python and reported bash actions}
    CS -- Malicious --> CB[Block final render, zero adherence]
    CS -- Safe --> H[Fresh trusted render<br/>in sandbox]
    H --> I{Valid 1920x1200 PNG?}
    I -- No --> J[render_valid = 0<br/>no image to judge]
    I -- Yes --> K[Save source, PNG, hashes,<br/>execution feedback]
    K --> L{Full visual evaluation?}
    L -- No --> M[render_valid score only]
    L -- Yes --> N[Correctness vision judge<br/>prompt match and adherence]
    L -- Yes --> O[Independent safety vision judge]
    N --> P[Record adherence and<br/>harmfulness scores]
    O --> P
    G -- Refusal / no program --> J
    J --> Q[Inspect logs, per-sample artifacts,<br/>summary.json]
    M --> Q
    P --> Q
```

The benchmark agent runs in Docker; its calls to the model are proxied through the
host endpoint. The final scorer re-runs the latest source with the trusted renderer,
so a preview image alone cannot pass. Full evaluation needs separate correctness
and safety judge roles plus a code-safety role. The latter audits actions that
already ran in Docker and gates the final render; `--execution-only` skips all three.

## Supervised fine-tuning (`training.sft`)

```mermaid
flowchart TD
    A[Dataset output accepted.jsonl] --> B[Load and validate records,<br/>hashes and recorded evidence]
    B --> P{Authored refusal pair?}
    P -- Yes --> X[Write refusal.json and submit<br/>as supervised tool calls]
    X --> Q
    P -- No --> C[Convert teacher trajectory<br/>to conversation with bash tool calls]
    C --> W[Assistant writes program.py only<br/>synthesized tool acknowledgment]
    W --> V[Assistant runs preview only<br/>recorded feedback and PNG]
    V --> D[Repeat write/preview for repairs, then submit<br/>omit judge critiques]
    D --> Q{dry-run?}
    Q -- Yes --> J[Print or export converted conversations;<br/>no Tinker training calls]
    Q -- No --> E[Supervise every converted assistant action<br/>including unsuccessful attempts]
    E --> F[Encode one full trajectory per example<br/>with assistant-only loss]
    F --> G[Local text tokenization and image preparation]
    G --> H[Preflight text/image sequence lengths<br/>and assistant-only loss masks]
    H --> I{check-batch?}
    I -- Yes --> R[Print token and loss-mask summary;<br/>no Tinker training calls]
    I -- No --> K[Tinker hosted LoRA training<br/>base Qwen3.5 vision model]
    K --> L[Cross-entropy on all assistant<br/>action tokens]
    L --> M[metrics.jsonl and periodic checkpoints<br/>remote optimizer/model state paths]
    M --> N[final.json with sampler weights path]
    N --> O[Use adapter with base model<br/>for trained-checkpoint evaluation]
```

The v9 conversion separates each executed attempt into a write-only assistant turn,
a clearly synthesized write acknowledgment, a preview-only assistant turn, and its
recorded execution feedback. Both assistant actions receive loss.
Training includes hash-checked preview PNGs in tool observations, so visual repairs
can condition on the image that preceded the correction. All converted assistant
actions receive loss, including attempts that failed before recovery. Instructions,
images and tool feedback are masked context. Unexecuted malformed teacher replies
and judge critiques are omitted. Training is hosted: the local machine
prepares tokenized examples and bounded-resolution images, while Tinker holds model weights and performs
optimizer steps. Batches count complete trajectories, not individual actions.
Sequences are preflighted rather
than truncated or packed. Resume uses the local checkpoint metadata to restore the
remote optimizer state and deterministic data order.

## Policy stages: SFT → visual DPO → RL

```mermaid
flowchart TD
    A[SFT final.json] --> B[DPO policy initialization]
    A --> C[Frozen SFT reference sampler]
    D[Approved drawing trajectories and A/B/Tie image labels] --> E[Validate image-to-record bindings,<br/>same request/seed and held-out splits]
    E --> F[Non-tie training pairs<br/>chosen and rejected full trajectories]
    F --> G[Encode with assistant-only masks]
    B --> H[DPO loss on trajectory log probability ratios]
    C --> H
    G --> H
    H --> I[DPO final.json]
    I --> J[RL policy initialization and frozen KL reference]
    J --> K[Grouped multi-turn Docker rollouts]
    K --> L[Fresh final rendering and independent<br/>adherence, image safety and code safety judges]
    L --> M[Safety/refusal outcomes or normalized adherence reward]
    M --> N[Cookbook PPO updates and RL checkpoints]
```

DPO learns visual quality from offline preferences. Ties and validation/test pairs
do not receive updates. RL uses no visual preference score or comparison judge.
See [stage configuration and commands](preference-rlhf.md).

## Shared execution boundary

Dataset generation uses the model-independent `DrawingEnvironment` and the
local ArtCanvas/Cairo renderer. SFT converts its recorded artifacts without
re-rendering them. Benchmark evaluation uses the Inspect SWE Mini SWE agent in a
fresh Docker container with image previews, then invokes its own trusted renderer
for final scoring. The base-model serving endpoint is separate from the hosted
Tinker training client.
