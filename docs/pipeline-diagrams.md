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
    C --> D[Local execution and PNG render]
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
    S -- Yes --> T[Accept and deduplicate]
    S -- No --> N
    U[Refusal] --> N
    T --> V[Train, validation, test JSONL]
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

## Evaluation (`evaluation.evaluate`)

```mermaid
flowchart TD
    A[evaluation/prompts.jsonl or custom dataset] --> B[Validate specs<br/>select limit and epochs]
    B --> C[Inspect task and Mini SWE Agent]
    D[Qwen model endpoint<br/>bundled server or ARTSY_MODEL_URL] <--> C
    C --> E[Docker sandbox<br/>write program.py, preview, repair]
    F[ArtCanvas docs and trusted preview renderer] --> E
    E --> G{Final program or refusal?}
    G -- Program --> H[Fresh trusted render<br/>in sandbox]
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
and safety judge roles; `--execution-only` skips both.

## Supervised fine-tuning (`training.sft`)

```mermaid
flowchart TD
    A[Dataset output accepted.jsonl] --> B[Load and validate records,<br/>hashes, review status]
    B --> P{Authored refusal pair?}
    P -- Yes --> X[Write refusal.json and submit<br/>as supervised tool calls]
    X --> Q
    P -- No --> C[Convert teacher trajectory<br/>to conversation with bash tool calls]
    C --> D[Use code and textual execution feedback<br/>as context; omit images and judge critiques]
    D --> Q{dry-run?}
    Q -- Yes --> J[Print or export converted conversations;<br/>no Tinker training calls]
    Q -- No --> E[Select successful program or refusal<br/>action and submission for loss]
    E --> F[Build one prefix per selected action<br/>with Qwen3.5 renderer]
    F --> G[Local text tokenization]
    G --> H[Preflight sequence lengths,<br/>text-only inputs and loss masks]
    H --> I{check-batch?}
    I -- Yes --> R[Print token and loss-mask summary;<br/>no Tinker training calls]
    I -- No --> K[Tinker hosted LoRA training<br/>base Qwen3.5 vision model]
    K --> L[Cross-entropy on selected assistant<br/>action tokens only]
    L --> M[metrics.jsonl and periodic checkpoints<br/>remote optimizer/model state paths]
    M --> N[final.json with sampler weights path]
    N --> O[Use adapter with base model<br/>for trained-checkpoint evaluation]
```

Training uses text-only conversations: recorded PNGs are checked for artifact
integrity but excluded from model inputs. Training is hosted: the local machine
prepares tokenized examples, while Tinker holds model weights and performs
optimizer steps. Sequences are preflighted rather
than truncated or packed. Resume uses the local checkpoint metadata to restore the
remote optimizer state and deterministic data order.

## Shared execution boundary

Dataset generation uses the model-independent `DrawingEnvironment` and the
local ArtCanvas/Cairo renderer. SFT converts its recorded artifacts without
re-rendering them. Benchmark evaluation uses the Inspect SWE Mini SWE agent in a
fresh Docker container with image previews, then invokes its own trusted renderer
for final scoring. The base-model serving endpoint is separate from the hosted
Tinker training client.
