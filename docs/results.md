# Full results

All denominators include invalid, truncated and rejected outputs. `model` = strict two-object output; `D` = deterministic conversion of predicted fields. A = 1e-4; B = 5e-5.

## Development controls

| Untrained control | C /60 | Q /48 | F /60 ↓ | P /48 ↓ | U /12 | Total /120 |
|---|---:|---:|---:|---:|---:|---:|
| internal | 54 | 1 | 1 | 38 | 11 | 66 |
| visible | 0 | 0 | 0 | 0 | 0 | 0 |
| visible_D | 37 | 0 | 0 | 45 | 0 | 37 |

The internal control uses one final JSON object. The visible control asks the base model for two objects; all 120 fail that strict protocol, so its zero score must not be presented as the general capability of the base model. Conversion of its first object is scored separately. Both controls have the same cases as the epoch experiment.

## All epochs and both splits

| Arm | Epoch | Split | Mode | C /60 | Q /48 | F /60 ↓ | P /48 ↓ | U /12 | Total /120 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| A | 1 | dev | model | 44 | 34 | 13 | 1 | 8 | 86 |
| A | 1 | dev | D | 54 | 37 | 3 | 2 | 7 | 98 |
| A | 1 | seen | model | 53 | 33 | 7 | 8 | 12 | 98 |
| A | 1 | seen | D | 57 | 38 | 1 | 4 | 12 | 107 |
| A | 2 | dev | model | 35 | 37 | 24 | 0 | 10 | 82 |
| A | 2 | dev | D | 53 | 40 | 5 | 0 | 10 | 103 |
| A | 2 | seen | model | 52 | 42 | 8 | 1 | 12 | 106 |
| A | 2 | seen | D | 57 | 45 | 2 | 1 | 12 | 114 |
| A | 3 | dev | model | 53 | 44 | 4 | 0 | 10 | 107 |
| A | 3 | dev | D | 53 | 46 | 4 | 0 | 10 | 109 |
| A | 3 | seen | model | 56 | 47 | 3 | 0 | 12 | 115 |
| A | 3 | seen | D | 56 | 48 | 2 | 0 | 12 | 116 |
| B | 1 | dev | model | 26 | 35 | 31 | 0 | 9 | 70 |
| B | 1 | dev | D | 47 | 37 | 11 | 0 | 6 | 90 |
| B | 1 | seen | model | 43 | 39 | 16 | 3 | 12 | 94 |
| B | 1 | seen | D | 52 | 43 | 6 | 1 | 12 | 107 |
| B | 2 | dev | model | 31 | 42 | 27 | 1 | 10 | 83 |
| B | 2 | dev | D | 56 | 45 | 2 | 1 | 9 | 110 |
| B | 2 | seen | model | 51 | 43 | 9 | 2 | 12 | 106 |
| B | 2 | seen | D | 58 | 46 | 1 | 1 | 12 | 116 |
| B | 3 | dev | model | 49 | 43 | 10 | 0 | 9 | 101 |
| B | 3 | dev | D | 56 | 46 | 2 | 0 | 9 | 111 |
| B | 3 | seen | model | 56 | 43 | 3 | 4 | 12 | 111 |
| B | 3 | seen | D | 56 | 47 | 2 | 1 | 12 | 115 |

## Fixed-set teacher-forced loss

| Arm | Epoch | Seen loss | Dev loss |
|---|---:|---:|---:|
| A | 1 | 0.012880 | 0.023434 |
| A | 2 | 0.005891 | 0.017387 |
| A | 3 | 0.002972 | 0.008223 |
| B | 1 | 0.015675 | 0.030193 |
| B | 2 | 0.008077 | 0.020485 |
| B | 3 | 0.004983 | 0.011726 |

Teacher loss conditions on gold histories. It is auxiliary and does not replace free-generation accuracy. The raw teacher token arrays remain in the local archive; this public bundle includes their aggregate curves, not a CPU recomputation of those arrays.

## Paired comparisons

- A3 versus A1, model output: 28 newly correct cases, 7 newly wrong, net +21.
- B3 versus B1, model output: 32 newly correct, 1 newly wrong, net +31.
- B3 versus A3, converted system: 5 newly correct, 3 newly wrong, net +2.

## Inspectable examples

### ablation-dev-get_metric_series-09-complete

Open the recorded episode_return sequence belonging to r1f7b16; keep only the first one point.

Gold action:
```json
{"action": "call", "name": "get_metric_series", "arguments": {"run_id": "r1f7b16", "metric": "episode_return", "max_points": 1}}
```

**A-1**
```json
{"tool":"get_metric_series","fields":{"run_id":"r1f7b16","metric":"episode_return","max_points":1}}
{"action":"call","name":"get_metric_series","arguments":{"run_id":"r1f7b16","metric":"episode_return","max_points":1}}
```

**A-2**
```json
{"tool":"get_metric_series","fields":{"run_id":"r1f7b16","metric":"episode_return","max_points":1}}
{"action":"clarify","missing_fields":["max_points"]}
```

**A-3**
```json
{"tool":"get_metric_series","fields":{"run_id":"r1f7b16","metric":"episode_return","max_points":1}}
{"action":"call","name":"get_metric_series","arguments":{"run_id":"r1f7b16","metric":"episode_return","max_points":1}}
```

### ablation-dev-rank_runs-05-natural

My requested output is IDs sorted on peak_return in descending order, considering project boreal.

Gold action:
```json
{"action": "clarify", "missing_fields": ["environment", "limit"]}
```

**A-1**
```json
{"tool":"rank_runs","fields":{"project":"boreal","environment":null,"metric":"peak_return","order":"desc","limit":null}}
{"action":"clarify","missing_fields":["limit"]}
```

**A-2**
```json
{"tool":"rank_runs","fields":{"project":"boreal","environment":null,"metric":"peak_return","order":"desc","limit":null}}
{"action":"clarify","missing_fields":["environment","limit"]}
```

**A-3**
```json
{"tool":"rank_runs","fields":{"project":"boreal","environment":null,"metric":"peak_return","order":"desc","limit":null}}
{"action":"clarify","missing_fields":["environment","limit"]}
```

### ablation-dev-get_run_config-05-complete

The training setup attached to run r1f6d02 is what I need.

Gold action:
```json
{"action": "call", "name": "get_run_config", "arguments": {"run_id": "r1f6d02"}}
```

**A-1**
```json
{"tool":"get_run_config","fields":{"run_id":"r1f6d02"}}
{"action":"call","name":"compare_algorithms","arguments":{"project":"lyra","environment":"Hopper-v4","metric":"final_return","algorithms":["SAC","PPO"]}}
```

**A-2**
```json
{"tool":"get_run_config","fields":{"run_id":"r1f6d02"}}
{"action":"call","name":"get_run_config","arguments":{"run_id":"r1f6d02"}}
```

**A-3**
```json
{"tool":"get_run_config","fields":{"run_id":"r1f6d02"}}
{"action":"call","name":"get_run_config","arguments":{"run_id":"r1f6d02"}}
```

### ablation-dev-summarize_metric-04-complete

Use final_return from matching runs to obtain its minimum, considering project quartz and environment Hopper-v4 and SAC.

Gold action:
```json
{"action": "call", "name": "summarize_metric", "arguments": {"project": "quartz", "environment": "Hopper-v4", "algorithm": "SAC", "metric": "final_return", "stat": "min"}}
```

**A-1**
```json
{"tool":"summarize_metric","fields":{"project":"quartz","environment":"Hopper-v4","algorithm":"SAC","metric":"final_return","stat":"min"}}
{"action":"call","name":"summarize_metric","arguments":{"project":"quartz","environment":"Hopper-v4","algorithm":"SAC","metric":"final_return","stat":"min"}}
```

**A-2**
```json
{"tool":"summarize_metric","fields":{"project":"quartz","environment":"Hopper-v4","algorithm":"SAC","metric":"final_return","stat":"min"}}
{"action":"call","name":"summarize_metric","arguments":{"project":"quartz","environment":"Hopper-v4","algorithm":"SAC","metric":"final_return","stat":"min"}}
```

**A-3**
```json
{"tool":"summarize_metric","fields":{"project":"quartz","environment":"Hopper-v4","algorithm":"SAC","metric":"final_return","stat":null}}
{"action":"clarify","missing_fields":["stat"]}
```

### ablation-dev-compare_algorithms-02-complete

For PPO and SAC, calculate a mean peak_return for each, considering nova project, Hopper-v4.

Gold action:
```json
{"action": "call", "name": "compare_algorithms", "arguments": {"project": "nova", "environment": "Hopper-v4", "metric": "peak_return", "algorithms": ["PPO", "SAC"]}}
```

**A-1**
```json
{"tool":"compare_algorithms","fields":{"project":"nova","environment":"Hopper-v4","metric":"peak_return","algorithms":["PPO","SAC"]}}
{"action":"call","name":"compare_algorithms","arguments":{"project":"nova","environment":"Hopper-v4","metric":"peak_return","algorithms":["PPO","SAC"]}}
```

**A-2**
```json
{"tool":"compare_algorithms","fields":{"project":"nova","environment":"Hopper-v4","metric":"peak_return","algorithms":["PPO","SAC"]}}
{"action":"clarify","missing_fields":["run_id"]}
```

**A-3**
```json
{"tool":"list_runs","fields":{"project":"nova","environment":"Hopper-v4","algorithm":"PPO"}}
{"tool":"list_runs","fields":{"project":"nova","environment":"Hopper-v4","algorithm":"SAC"}}
{"action":"call","name":"compare_algorithms","arguments":{"project":"nova","environment":"Hopper-v4","metric":"peak_return","algorithms":["PPO","SAC"]}}
```

The first cases show recovered field/action consistency and exact missing sets. The final cases preserve remaining failures: a lexical mapping problem (`minimum` → `min`) and extra field objects that violate the strict protocol. No relaxed rescoring is used for the headline.

Machine-readable metrics: [epochs.json](../results/epochs.json). Raw outputs: [predictions/](../results/predictions/).
