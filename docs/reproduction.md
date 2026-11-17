# Reproduction and validation

## 1. Re-score existing predictions (CPU)

Use the quickstart in [README](../README.md). Python 3.11 and `requirements-cpu.txt` (jsonschema and NumPy) are sufficient. `fieldfirst.verify` validates release hashes, case order, synthetic labels, SFT targets and generation prompt hashes, then reruns the original strict scorer and deterministic converter. It recomputes all epoch-cell and base-control scores, including error counts and per-case model-to-converter wins/losses. Cross-checkpoint contrast summaries, teacher-loss aggregates and training events are hash-checked exports; they are not recomputed by this command.

This reproduces **scores from observed outputs**. It does not independently regenerate outputs from model weights.

## 2. Train from the pinned base (Apple Silicon)

The original runs used MLX on a local Apple Silicon Mac. Windows/CUDA is not implemented. Model and adapter files are deliberately excluded from Git.

```bash
# Run from the repository root. Use a separate environment from CPU-only verification.
python3.11 -m venv "$HOME/.local/share/field-first-tool-calling/venv-mlx"
source "$HOME/.local/share/field-first-tool-calling/venv-mlx/bin/activate"
python -m pip install -r requirements-macos-arm64.txt

# Bounded installation and checkpoint smoke test, not an accuracy experiment.
python -B -m fieldfirst.train --arm A --name smoke-A --smoke

# Run sequentially: both arms start from the same base and seed.
python -B -m fieldfirst.train --arm A --name lr1e-4-three-epochs
python -B -m fieldfirst.train --arm B --name lr5e-5-three-epochs
```

Default runtime: `~/.local/share/field-first-tool-calling/`. Set `--runtime /path/outside/repository` or `TOOLCALL_RUNTIME_DIR` to change it. The runner rejects runtime paths inside or above the source tree and symlinks redirecting there. Model download, environment, cache and checkpoints belong outside the checkout. `--snapshot /path/to/model` uses an existing pinned snapshot; model/tokenizer file hashes are checked against [model-files.json](../data/model-files.json).

A full run consumes 600 examples three times. It writes:

- `config.json`: recipe and run identity.
- `events.jsonl`: loss, supervised tokens, timing and MLX memory every eight microsteps.
- `step-0200/` through `step-1800/`: adapters, optimizer, RNG and verified metadata.
- `eval-0600/`, `eval-1200/`, `eval-1800/`: seen/dev raw generations, teacher-forced token diagnostics and model/converter metrics.
- `resources-*.jsonl`, `resource-summary.json`: resource guard evidence.

Rerun **the identical command** to resume from the last complete checkpoint. Changed recipes are rejected. An exclusive run lock prevents simultaneous writers. Training guards stop if sampled process RSS exceeds 10 GiB or system swap grows over 2 GiB during the process. Available system memory matters; these thresholds are safeguards, not hardware requirements.

## Frozen recipe

| Parameter | Value |
|---|---|
| Base | `mlx-community/Qwen3-1.7B-4bit` |
| Hugging Face revision | `3b1b1768f8f8cf8351c712464f906e86c2b8269e` |
| Training examples | Original 600, same order repeated each epoch |
| Targets | Required fields with value/null, newline, final action JSON |
| Seed | 20261113 |
| LoRA | Last 8 layers; q/v projections; rank 8; scale 20; dropout 0 |
| Trainable parameters | 458,752 |
| Loss | Assistant-only token CE, mean within each example, then mean across 8 examples |
| Optimizer | Adam, betas (0.9, 0.999), epsilon 1e-8, **bias correction disabled** |
| Learning rate | A: 1e-4; B: 5e-5; constant, no scheduler |
| Batch / accumulation | 1 / 8 |
| Epoch budget | 600 microsteps, 75 updates, 29,718 supervised tokens |
| Total per arm | 1,800 microsteps, 225 updates |
| Training length cap | 1,217 tokens; truncation is prohibited |
| Generation | Temperature 0, thinking disabled, max 384 generated tokens |
| Teacher diagnostic cap | 2,048 tokens |

Bias correction is deliberately disabled to preserve the tested recipe; it is not a general recommendation for Adam. Teacher loss uses token averaging over a fixed dataset and gold histories; online loss averages examples over a changing training window. These quantities should not be compared as identical objectives.

## Validation scope

The original complete experiment ran and verified all 12 free-generation cells and 12 teacher-loss cells. Arm A reused its original epoch-1 checkpoint and continued the full optimizer/RNG state; arm B started fresh. The five newly trained epochs took about 121 minutes in summed training-step time, excluding evaluation and load time. Peak MLX allocation was about 3.65 GB; it does not represent total system memory.

The public wrapper starts both arms fresh and retains the original numerical primitives. It was checked with CPU tests (with optional Metal and three archive-dependent integration tests explicitly skipped) and a real eight-microstep Apple Silicon smoke test, checkpoint save/reload, four free generations and four teacher-forced examples. Its first update reproduced the original loss exactly: `0.8615061156451702`, with 360 supervised tokens. Full three-epoch training has **not been rerun through this new wrapper**. Different MLX/hardware versions may change outputs; do not assume bitwise reproduction on another machine.

Original model and optimizer checkpoints are not part of this release. The original orchestration modules remain as source evidence under `experiments/` and `toolcall_lab/`; their historical `--stage` commands require archived parent-run directories and review documents. Use `fieldfirst.train` for a fresh machine.

## Source and artifact provenance

[provenance.json](../results/provenance.json) records the original source hashes and the hashes of the pre-export raw generation files. Published generation rows preserve IDs, exact output text, finish reason and prompt hash when recorded; timing fields are omitted. [manifest.json](../results/manifest.json) binds the public data and metrics. These hashes support integrity checks, not independent replication of model training.

Base model: [pinned Hugging Face snapshot](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit/tree/3b1b1768f8f8cf8351c712464f906e86c2b8269e). Frameworks: [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm). Model terms apply independently of this code repository.

To regenerate the two figures, install Matplotlib in a separate plotting environment and run `python -B -m fieldfirst.plot`. Figure inputs are the published counts and loss aggregates.

## GitHub Actions status

The initial public release includes [a verification workflow template](verify-workflow.yml). It is not enabled automatically: the publishing GitHub CLI credential lacks the `workflow` scope, and the connected integration could not write the workflow. Local fresh-environment CPU tests and rescoring passed as recorded in [publication-validation.json](../results/publication-validation.json). To enable CI with appropriate GitHub permissions, copy the template to `.github/workflows/verify.yml`.
