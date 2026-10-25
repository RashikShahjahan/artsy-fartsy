# Datagen, evaluation, and training flows

These diagrams describe the repository's current Python pipeline. They distinguish
dataset-generation review from benchmark evaluation: the former produces candidate
training data, while the latter measures a model on a fixed test set.

## Dataset generation (`drawing.generate_dataset`)

```mermaid
flowchart TD
    A[CLI options and prompt source] --> B[Generate or reuse prompt plan]
    B --> AA{Harmful refusal slot?}
    AA -- Yes --> AB[Prompt writer supplies refusal target]
    AB --> AC[Accepted refusal record, no render]
    AC --> V
    AA -- No --> C[Teacher writes ArtCanvas Python]
    C --> CS{Code-safety monitor}
    CS -- Malicious --> N[Reject before execution]
    CS -- Monitor error --> CE[Infrastructure error, no execution]
    CS -- Safe / unconfigured --> D[Local execution and PNG render]
    D --> E{Execution valid?}
    E -- No --> F[Send execution feedback]
    F --> C
    E -- Yes --> G{Both judges configured?}
    G -- No --> H[Fresh submission]
    G -- Yes --> I[Correctness vision judge]
    G -- Yes --> J[Independent safety vision judge]
    I --> K{Score passes and image safe?}
    J --> K
    K -- Low score --> M[Send visual critique]
    M --> C
    K -- Harmful --> N[Reject]
    K -- Pass --> H
    H --> O{Fresh render valid?}
    O -- No --> F
    O -- Yes --> P{Both judges configured?}
    P -- No --> Q[Review queue]
    P -- Yes --> R[Rejudge final PNG with both roles]
    R --> S{Score meets threshold and safe?}
    S -- Yes --> CP{Code review passed?}
    CP -- Yes --> T[Accept and deduplicate]
    CP -- Unconfigured --> Q
    S -- No --> N
    U[Refusal] --> N
    T --> V[Combined accepted JSONL]
```

Prompt planning can also be bypassed by supplying input specs. With
`--harmful-count`, the writer returns refusal-worthy prompts paired with target
refusals. These are accepted into training without rendering unsafe code and
must be audited before fine-tuning. Without both vision judges, executable
ordinary drawings remain in `review.jsonl`; unexpected refusals to ordinary
drawing requests are rejected because no final image can be graded. Both judge roles use the same image prompts
and JSON parsing as Inspect, and their requests run concurrently. Generation
checkpoints per candidate and writes its artifacts, trajectory, manifest, and
summary under the output directory.
`--malicious-code-count` adds a distinct refusal category for unsafe computing
actions. The code-safety monitor reviews every teacher source before execution;
automatic drawing acceptance requires it in addition to both vision judges.

## Evaluation (`evaluation.evaluate`)

```mermaid
flowchart TD
    A[evaluation/prompts.jsonl or custom dataset] --> B[Validate specs<br/>select limit and epochs]
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
    A[Dataset output accepted.jsonl] --> B[Load and validate records,<br/>hashes, review status]
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

## Shared execution boundary

Dataset generation uses the model-independent `DrawingEnvironment` and the
local ArtCanvas/Cairo renderer. SFT converts its recorded artifacts without
re-rendering them. Benchmark evaluation uses the Inspect SWE Mini SWE agent in a
fresh Docker container with image previews, then invokes its own trusted renderer
for final scoring. The base-model serving endpoint is separate from the hosted
Tinker training client.
