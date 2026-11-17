# Research path and what each comparison can tell us

The project began with schema changes and missing-information failures in local small-model tool calling. The released headline experiment focuses on missing information; it does not claim that the final adapter has been validated against a new interface-renaming test.

1. **Baselines and prompt controls.** Test complete, missing-information and unsupported tasks; strengthen instructions about grounding/defaults and inspect behavior before training. This establishes whether a model is guessing versus asking.
2. **Direct-action supervised fine-tuning.** Use a mixture of complete, missing and unsupported cases. Aggregate loss can fall while the decision policy shifts toward over-clarification. Training data therefore cannot be characterized as "all clarification examples."
3. **Paired data and learning-rate comparisons.** Remove fixed textual associations and inventory-order shortcuts; compare controlled variants. These experiments motivate matching exposure and inspecting paired behavior rather than reading a single score.
4. **Training mechanism controls.** Test sample-versus-token loss normalization and Adam bias correction separately. These are distinct interventions, not hidden changes in the final learning-rate comparison.
5. **Action-position weighting.** Increase supervision on the action decision instead of treating every output token as equally consequential. This probes the gap between token CE and whole-task correctness.
6. **Field-first supervision.** Supervise known/null values before the final action; separately score field extraction, the action and their consistency.
7. **Field-to-action diagnosis.** On the one-epoch model, raw model success was 86/120; deterministic conversion of its own fields was 98/120. Supplying gold fields and continuing the original prompt reached 103/120, but is an oracle diagnosis, not deployable accuracy. A separate mapping prompt scored 42/120, showing prompt/distribution changes can matter. These historical diagnostic totals are described here as context; their raw outputs are not included in the public CPU-rescoring grid.
8. **Longer training with two rates.** Hold the original 600 examples, targets, ordering and optimizer fixed. Evaluate every epoch at 1e-4 and 5e-5 on the same seen/dev samples. This is the main fully exported experiment.

Source for earlier experiments is retained, including tests, rather than claiming every historical run is reproducible from the compact public artifact bundle. The independent public training entrypoint implements the final recipe; the CPU release grid includes both base controls and all six epoch checkpoints on seen/dev.

## Why low token loss did not guarantee the right action

Teacher forcing supplies the correct earlier tokens. JSON syntax, repeated key names and copied field values can be easy, while a small number of `call`/`clarify`/`null` decisions dominate task success. One wrong decision can invalidate an otherwise correct response. Free generation also conditions on the model's own earlier mistakes. We therefore retain fixed-set teacher loss as a diagnostic and score full generated outputs separately.

A complete request for one metric point illustrates the mismatch: the model could produce `max_points: 1` correctly and then ask the user for `max_points` anyway. More training eventually corrected the action in this case. It would be misleading to interpret every loss reduction as a monotonic improvement in full-task accuracy.

## What has not been established

No test here proves that increasing dataset size helps. There is only one seed, no newly sealed test, no calibrated uncertainty estimate and no real deployment study. Further work should diversify the input distribution and schemas, freeze a fresh test set before model selection, and repeat training with several seeds. Gains of two cases should be treated as descriptive differences.
