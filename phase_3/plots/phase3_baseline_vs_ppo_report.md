# Phase‑3 Evaluation Report (Baseline vs PPO)

Date: 2026‑04‑20

## Dataset

- Directory: `phase_3/test_dataset`
- Matrix: scenarios **8, 9, 10** × patterns **B, C** × seeds **11, 12**  
  Total: **12 runs** (12 `*_tick.csv` files)
- Per run: `ueCount=8`, `duration=300s`, `tick=0.1s` (≈24,000 rows per file)

## Compared methods

- **Baseline**: Phase‑1 `Algorithm1` with static parameters: **TTT=250 ms**, **HYS=4.5 dB**
- **PPO agent**: Phase‑2 trained policy (`phase_2/models/final`) controlling `Algorithm1` via ΔTTT/ΔHYS actions (greedy argmax on actor logits)

## Overall results (sum over 12 files)

Lower is better for all three metrics.

| Metric | Baseline | PPO | PPO / Baseline |
|---|---:|---:|---:|
| Handovers (HOs) | 11,528 | 149,006 | 12.93× |
| Radio link failures (RLFs) | 30,066 | 38,358 | 1.28× |
| Ping‑Pongs | 432 | 7,310 | 16.92× |

Relative “improvement over baseline” (as written by the evaluator):

- HO reduction: **−1192.56%**
- RLF reduction: **−27.58%**
- Ping‑Pong reduction: **−1592.13%**

## Breakdown by scenario/pattern (sum over both seeds)

| Scenario | Pattern | Baseline HOs | PPO HOs | Baseline RLFs | PPO RLFs | Baseline PP | PPO PP |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | B | 1,308 | 9,660 | 2,074 | 2,102 | 40 | 678 |
| 8 | C | 694 | 8,770 | 962 | 1,116 | 26 | 724 |
| 9 | B | 4,566 | 33,316 | 4,780 | 5,974 | 184 | 2,872 |
| 9 | C | 4,720 | 28,146 | 5,828 | 5,536 | 182 | 2,512 |
| 10 | B | 136 | 38,814 | 7,472 | 12,314 | 0 | 196 |
| 10 | C | 104 | 30,300 | 8,950 | 11,316 | 0 | 328 |

## Output artifacts

- Summary JSON: `phase_3/plots/comparison_summary_v3.json`
- Per‑file metrics CSV: `phase_3/plots/per_file_metrics_v3.csv`
- Plot: `phase_3/plots/comparison_results_v3.png`
