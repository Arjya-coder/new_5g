# Phase‑3 Unseen Evaluation — Analysis Report (2026‑04‑20)

## What was evaluated
- Phase‑3 dataset: [phase_3/test_dataset](phase_3/test_dataset)
  - 45 runs = scenarios **8–10** × patterns **A/B/C** × seeds **11–15**
  - Size: **1349.58 MB**
- PPO model: [phase_2/models/final](phase_2/models/final)
- Evaluator: [phase_3/compare_models.py](phase_3/compare_models.py)
- Command used:
  - `python -m phase_3.compare_models --control-interval-steps 5`
- Outputs:
  - Plot: [phase_3/plots/comparison_results_v3.png](phase_3/plots/comparison_results_v3.png)
  - Per-file metrics CSV: [phase_3/plots/per_file_metrics_v3.csv](phase_3/plots/per_file_metrics_v3.csv)
  - Summary JSON: [phase_3/plots/comparison_summary_v3.json](phase_3/plots/comparison_summary_v3.json)

## Executive summary (why it looks “so bad”)
Across the full Phase‑3 dataset, the PPO policy performs significantly worse than the baseline on all metrics:

- Baseline totals: **HOs 232,870**, **RLFs 630,310**, **PingPongs 8,815**
- PPO totals:      **HOs 2,999,515**, **RLFs 844,380**, **PingPongs 145,715**

This is not a small regression — it is dominated by a **handover/ping‑pong explosion**, especially in **Scenario 10**.

## Where the failure happens (scenario breakdown)
The following totals are summed over all seeds 11–15 and patterns A/B/C.

| Scenario | Baseline HOs | PPO HOs | HO ratio | Baseline RLFs | PPO RLFs | RLF ratio | Baseline PP | PPO PP | PP ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 37,795 | 292,290 | 7.7× | 51,985 | 46,600 | 0.90× | 1,060 | 25,060 | 23.6× |
| 9 | 185,605 | 1,181,160 | 6.4× | 194,955 | 215,755 | 1.11× | 7,755 | 110,610 | 14.3× |
| 10 | 9,470 | 1,526,065 | 161× | 383,370 | 582,025 | 1.52× | 0 | 10,045 | n/a |

Key takeaway:
- Scenario 10 is the main collapse: PPO produces **~161× more handovers** than baseline and also increases RLF substantially.

## Important diagnostic findings
### 1) Seeds are not changing results
In [phase_3/plots/per_file_metrics_v3.csv](phase_3/plots/per_file_metrics_v3.csv), each `(scenario, pattern)` has identical metrics for seeds 11–15.

That strongly indicates the Phase‑3 generator is effectively deterministic and the `--seed` argument is **not** driving the random processes.

Why this matters:
- You cannot get robustness from “multiple seeds” if the seeds don’t produce different trajectories/interference.
- Any fine‑tuning you do on Phase‑3 will be less reliable.

### 2) RLF counting is “RLF‑ticks”, not “RLF events”
In [phase_3/compare_models.py](phase_3/compare_models.py), the code increments the RLF counter on every tick after SINR stays below ‑20 dB for ≥2 ticks.

That inflates absolute values, but PPO vs baseline comparison is still consistent because both use the same measurement.

## Likely root causes (most plausible)
1) **Out‑of‑distribution (OOD) shift**
   - Phase‑2 training data = scenarios 1–7.
   - Phase‑3 evaluation = scenarios 8–10 with different dynamics (ping‑pong stress, non‑stationary interference, coverage holes).
   The learned policy has no reason to behave well under those new dynamics.

2) **Policy learned “aggressive handovers” as a shortcut**
   The policy can reduce some link‑failure risk by making TTT/HYS small, which triggers handovers frequently.
   This can look acceptable in training scenarios but becomes catastrophic in Scenario 10 (coverage‑hole behavior).

3) **Handover logic + MUST_HO can amplify thrashing**
   In the Phase‑2 handover logic, when conditions force handovers, low TTT/HYS makes many candidates eligible quickly.
   Without a strong dwell/lockout, this yields fast back‑and‑forth switching (ping‑pong) and runaway HO counts.

## How to improve (actionable, prioritized)
### Priority A — Fix Phase‑3 seeding (required)
Update [phase_3/phase3_eval_scenarios.cc](phase_3/phase3_eval_scenarios.cc) so `--seed` actually changes randomness.

Expected outcome:
- Different seeds → different paths/jitter/interference → meaningful robustness testing.

### Priority B — Prevent HO/PingPong explosions (reward + safety constraints)
In [phase_2/algo_2.py](phase_2/algo_2.py):
- Increase HO and PingPong penalties in the reward.
- Increase HO-frequency penalty (penalize repeated HOs within a time window).
- Add/strengthen a dwell/lockout after a HO inside `Algorithm1` (suppress HO for e.g. 0.5–1.0s unless link is collapsing).

Expected outcome:
- PPO can’t “cheat” by doing excessive handovers.

### Priority C — If Phase‑3 performance is required: fine‑tune on Phase‑3
Treat Phase‑3 as a new target domain:
- Train seeds: 11–13
- Val: 14
- Test: 15

Start from the Phase‑2 model and fine‑tune with lower learning rates.

Expected outcome:
- Improves Scenario 9/10 behavior substantially.

### Priority D — Make Phase‑3 evaluation more diagnostic
Enhance [phase_3/compare_models.py](phase_3/compare_models.py):
- Add “Algo2 static” baseline (run `phase_2/algo_2.Algorithm1` with fixed TTT/HYS) to separate:
  - changes due to the RL policy vs
  - differences in baseline algorithm implementations.
- Count RLF on the *rising edge* (first tick when counter reaches 2) for clearer event counts.
- Log PPO’s chosen `(TTT,HYS)` histogram per scenario/pattern.

## Recommended next move
1) Fix seeding in the Phase‑3 generator.
2) Add safety constraints / stronger HO penalties.
3) Fine‑tune on Phase‑3 (seed split) and re-run [phase_3/compare_models.py](phase_3/compare_models.py).
