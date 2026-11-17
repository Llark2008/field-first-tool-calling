# Action-position weighting pilot

This isolated experiment tests whether emphasizing the first action-discriminating token improves complete-versus-missing-information decisions. It preserves the historical training code, data, exposure order, optimizer and inference settings.

For each answer:

```
L = 0.20 * CE(first action-discriminating token)
  + 0.80 * mean(CE(other supervised answer tokens))
```

The decision token follows `{"action":"`. Only `clar` receives decision weight for `clarify`; `ify` remains in the remainder. The prompt and padding are masked. Existing EOS and post-EOS newline supervision is preserved in both arms.

- **C0:** verified historical M0 control, reused without redundant retraining.
- **W20:** independently initialized Qwen3-1.7B 4bit, last-eight-layer q/v LoRA, rank8, lr1e-4, Adam without bias correction.
- **Budget:** same first600 examples, microbatch1 × accumulation8 =75updates. Checkpoints25/50/75.
- **Evaluation:**120 development examples at each checkpoint; fixed120 seen-training examples at75, separately evaluated by free generation and teacher forcing. No sealed test access.
- **Primary goal:** at75updates, C≥56/60, F≤3/60, boundary≥27/30, U≥11/12 and Q≥17/48. Full grid, training identity and clean final resource evidence are required before a verdict.

The runner uses the existing external environment and runtime under `~/.local/share/local-tool-calling`; no model, checkpoint, environment or large log belongs in this synchronized directory.

```bash
TOOLCALL_TEST_MLX=1 bash scripts/local.sh -m unittest discover -s experiments/action_weighting/tests -v
bash scripts/local.sh -m experiments.action_weighting.run --stage serve
bash scripts/local.sh -m experiments.action_weighting.run --stage freeze
bash scripts/local.sh -m experiments.action_weighting.run --stage run
```

Metal must be available to the process. Freeze requires the saved implementation review and historical experiment/replay bundles; this is a lineage-dependent local experiment, not yet a fresh-clone reproduction command. Sources, data, model, control checkpoint and reused output identities are validated. Resume uses the same configuration and persisted swap baseline.

The monitor binds only `127.0.0.1:49407`. Runtime logs/checkpoints remain external. `--stage report` verifies the completed report marker before regenerating a final report; incomplete experiments receive no joint verdict.

Small exports live in `reports/action-weighting-results.*`. The optional standalone plot script `scripts/plot_action_weighting.py` runs in an isolated plotting environment, not the training environment. Curves retain raw loss, action loss and weighted objective as distinct quantities. The120 seen-training cases are a fixed paired/tool-balanced subset, not an unbiased census of the600 examples.

See the frozen design in `docs/superpowers/specs/2026-11-15-action-weighting-design.md` and the execution ledger for review findings and verification. A failed W20 pilot does not disprove all weighting settings; the user's conditional next route is explicit field output before the final action JSON, under a separate protocol.
