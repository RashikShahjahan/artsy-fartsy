# Welcome examples

These are saved RLHF model outputs, copied without image edits. They ship with the client so the welcome screen works without the drawing service.

| Asset | Model | Source artifact |
| --- | --- | --- |
| `landscape.png` | `artsy/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery` | `episodes/ui/ff89bc69bbc3486d91a85280c3374e21/output/image.png` |
| `fractal.png` | `artsy/rlhf-experiment10-genai1000-linear-epoch1-kl-budget-recovery` | `episodes/fractal-four-models-20261114-latest/rlhf-run/artifacts/fractal-1/image.png` |
| `metropolis.png` | `artsy/rlhf-genai300-linear-epoch1` | `episodes/metropolis-four-models-20261113/rlhf/artifacts/metropolis-1/image.png` |

Model provenance is recorded in the landscape run’s `job.json` and the evaluation runs’ `summary.json` files. Source episodes are local artifacts and are not required at runtime. The landscape was generated in response to “hi”; its example prompt describes the resulting scene.
