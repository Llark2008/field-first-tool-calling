# Data and evaluation contract

The release contains synthetic requests for experiment-tracking operations: listing/ranking runs, retrieving a run configuration or metric series, summarizing a metric, and comparing algorithms. IDs, projects, runs and results are generated fixtures, not user records. Tool schemas and fixtures are included so that calls can be validated and executed deterministically during scoring.

| File | Rows | Complete | Missing arguments | Unsupported |
|---|---:|---:|---:|---:|
| `train-cases.jsonl` | 600 | 300 | 240 | 60 |
| `sft-F.jsonl` | 600 | Same training cases, exact original order and targets |||
| `seen-cases.jsonl` | 120 | 60 | 48 | 12 |
| `dev-cases.jsonl` | 120 | 60 | 48 | 12 |

`seen` is a fixed subset of training, not a held-out set and not a measurement of all 600 training examples. `dev` was repeatedly consulted throughout research. Train and dev IDs differ, but this does not establish semantic independence or broad generalization: both are generated from a narrow task family.

Each case includes `request`, `context`, `tools` and evaluation-only annotations (`gold`, `supplied_arguments`, `missing_fields`, `fixture`, etc.). **Only request, context and tools enter the inference prompt.** Training targets use labels, as intended for supervised learning. Tests verify that removing every evaluation label leaves the inference prompt unchanged.

Complete/missing pairs share a task intention while required information is omitted in the missing version. The context may list known IDs, algorithms or environments, but explicitly says these inventories are not defaults or established user choices. Required fields are supervised as known values or JSON `null`. Optional fields are included only when supplied, apart from documented defaults.

The original construction and audits are retained in `toolcall_lab/formal_data.py`, `toolcall_lab/ablation_data.py`, `experiments/fields_sft/data.py` and their tests. The public training path consumes the exact exported 600 rows rather than regenerating a potentially different sample.

## Strict scoring

- **Model success:** exactly two JSON objects, newline-separated, with no extra prose/objects, followed by a valid final action that matches the request. Correct final actions embedded in malformed output do not pass.
- **Complete success (C):** correct tool and arguments, with successful fixture execution matching the target.
- **Missing success (Q):** `clarify` with the exact missing-field set. Asking for only one of two missing fields fails.
- **False clarification (F):** any actual `clarify` action on a complete case. Invalid output is a separate failure, so a zero F does not by itself indicate competence.
- **Premature call (P):** an actual call on a missing-information case.
- **Unsupported success (U):** correct `unsupported` action.
- **D:** parse the predicted first object, validate it against the public tool schema and deterministically choose the action. All rejected cases stay in the denominator. This can accept a valid first object even if the model later violates its two-object protocol; it is a separate system score.

The deterministic converter accepts schemas and predicted fields only, not cases/gold labels. It cannot fix wrong non-null field values. It rejects unknown tools/fields, missing required field keys, wrong types, duplicate keys, nonfinite JSON and truncated generations.
