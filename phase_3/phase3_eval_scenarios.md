# Phase-3 ns-3 evaluation scenarios (unseen)

This folder contains a Phase-3 ns-3 generator that produces **new/unseen** traces (not used in Phase-2 PPO training), while keeping the **same CSV schema** as `dataset.cpp`.

## Generator

- C++ program: [phase_3/phase3_eval_scenarios.cc](phase3_eval_scenarios.cc)
- Outputs (same naming convention as phase-1):
  - `<outputPrefix>_tick.csv`
  - `<outputPrefix>_events.csv`
  - `<outputPrefix>_summary.json`

## Scenarios

All scenarios export **identical** tick columns (`serving_*`, `best_neighbor_*`, `candidate_cell`, `a3_hold_ms`, `n1..n6`) so Phase-2 offline evaluators can replay them.

- **Scenario 8: HetNet Hotspot (Macro + Micros)**
  - One macro at the center + dense micro grid.
  - Designed to stress **ping-pong / stability** tradeoffs.
  - Patterns:
    - A: perimeter walk loop
    - B: in → roam → out commute
    - C: stop-go crowd cross

- **Scenario 9: Urban Grid + Interference Burst**
  - Macro grid + small-cells on a corridor.
  - Has a **time-bounded interference burst** (downtown box, ~180–240s) to create **non-stationary RF**.
  - Patterns:
    - A: diagonal drive
    - B: downtown U-turn loops
    - C: bus with stops

- **Scenario 10: Rural High-Speed Corridor + Coverage Hole**
  - Sparse corridor deployment.
  - Includes a **coverage hole** region (x≈3600–4400m, |y|≤60m) to stress **RLF + recovery**.
  - Patterns:
    - A: high-speed train (≈70 m/s)
    - B: variable-speed car
    - C: emergency U-turn crossing the hole

## How to run (ns-3)

1) Copy [phase_3/phase3_eval_scenarios.cc](phase3_eval_scenarios.cc) into your ns-3 `scratch/` folder.

2) Run (from ns-3 root):

```bash
./ns3 run "scratch/phase3_eval_scenarios --scenarioId=8 --pattern=A --seed=11 --ueCount=20 --outputPrefix=phase3_eval/s8_pA_seed11"
```

3) Collect outputs:
- `phase3_eval/s8_pA_seed11_tick.csv`
- `phase3_eval/s8_pA_seed11_events.csv`
- `phase3_eval/s8_pA_seed11_summary.json`

## Batch runner (Windows → WSL)

If you want the full `{8,9,10}×{A,B,C}×seeds` matrix automatically, use:

- Python runner: [phase_3/run_phase3_dataset.py](run_phase3_dataset.py)
- Output folder (only): `phase_3/test_dataset/`

Example (PowerShell from this repo root):

```powershell
python phase_3/run_phase3_dataset.py --seedStart 11 --seedEnd 15
```

### Quick preset (recommended for fast local testing)

If you want a **much faster** dataset that still covers the important Phase-3 stresses (8/9/10) without generating the full 45 runs, use:

```powershell
python phase_3/run_phase3_dataset.py --preset quick --clean
```

This generates a smaller matrix (by default):
- Scenario IDs: `8, 9, 10`
- Patterns: `B, C`
- Seeds: `11..12`
- `ueCount=8`, `duration=300s`

This runner copies the C++ file into WSL `ns-3` scratch and runs ns-3 in WSL, but writes the resulting CSV/JSON files directly into the Windows workspace under `phase_3/test_dataset`.

## Recommended evaluation matrix

To keep Phase-3 clearly “unseen”, use **new seeds** (e.g. 11–15).

- Scenario IDs: `8, 9, 10`
- Patterns: `A, B, C`
- Seeds: `11..15`

That yields `3 × 3 × 5 = 45` runs.
