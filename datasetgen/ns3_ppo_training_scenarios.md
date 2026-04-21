# NS‑3 Scenario Spec for “Best‑Possible” PPO Training (5G Handover)

Date: 2026‑04‑19

This document defines a **training scenario suite** (what to simulate in ns‑3, how to mix scenarios/patterns/seeds, and how to split train/val/test) that is designed to produce a PPO agent that:

- reduces **RLF** risk,
- avoids **ping‑pong** oscillation,
- and does not inflate **handover count** relative to a static baseline.

It is grounded in the current scenario generator in [dataset.cpp](dataset.cpp) (scenario IDs 1–7, patterns A/B/C).

---

## Does this tackle the previous Phase‑3 problems?

Partially — **by design**.

This spec tackles the *data/scenario* side of the root causes (coverage + balanced sampling) and documents the **must-fix offline ingestion requirements** that otherwise invalidate PPO training.

It does **not** automatically fix PPO design issues (reward/action/state/control‑rate). Those are listed as prerequisites in §0.2 because you must implement them in code (Phase‑2) before retraining.

---

## 0) Critical prerequisites (so scenarios actually train the right behavior)

Even a perfect ns‑3 dataset won’t help if the offline environment **distorts** the measurements.

Before training, ensure your offline env uses the ns‑3 CSV correctly.

### 0.1 Offline ingestion must be “cell‑consistent”

Your algorithm’s current serving cell can diverge from the generator’s logged `serving_cell` (because your Python `Algorithm1` may make different HO choices). Therefore:

- Serving metrics (**RSRP/SINR/distance**) must be reconstructed from the **neighbor columns** (`n1..n6`) for the *current* `algo1.serving_cell_id`.
- Only use `serving_rsrp_dbm` / `serving_sinr_db` when `row['serving_cell'] == algo1.serving_cell_id`. Otherwise those `serving_*` fields belong to the generator’s baseline, not your current connection.
- Serving distance must come from the matching `n{i}_d_m` entry; do **not** default to a constant distance (this collapses the “edge/coverage” signal).
- Do **not** floor/clamp SINR at \(-20\) dB if your RLF detector is `sinr < -20` (current Phase‑2 `TrainingEnv` does this). Clamping erases RLF events during training.

The current Phase‑2 offline wrapper that needs these fixes is [phase_2/train_rl.py](phase_2/train_rl.py).

Otherwise the agent won’t experience realistic RLF pressure and will over‑handover.

### 0.2 PPO‑interface prerequisites (these were major causes of the bad results)

These are not “scenario” problems, but without fixing them, *no* scenario suite will reliably produce a good policy:

- **Hidden control state**: your PPO actions are *deltas* of TTT/HYS (see [phase_2/algo_2.py](phase_2/algo_2.py)), but the state vector does not include the current TTT/HYS → partial observability → unstable parameter oscillation. Fix by either (a) adding current TTT/HYS to state, or (b) switching to an absolute action space.
- **Control rate too fast**: changing TTT/HYS every 100 ms allows high‑frequency thrashing. Fix by holding parameters for N ticks (e.g., 5–10 ticks) or stepping the RL control interval slower than the measurement tick.
- **Reward offset cancels HO cost**: a constant positive offset can negate HO penalties and make ping‑pong “cheap”. Ensure HO and ping‑pong penalties cannot be washed out by a baseline positive term.

---

## 1) What a “perfect training dataset” must contain

A good PPO handover controller must generalize across **four axes**:

1) **Speed regimes**
   - pedestrian: 1–2 m/s
   - fast walk / jog: 5 m/s
   - vehicle: 10–20 m/s
   - high speed: 25–35 m/s
   - variable speed profiles (stop‑go, acceleration cycles)

2) **Topology regimes**
   - dense multi‑neighbor urban grid (many strong neighbors)
   - canyon (correlated blockage, narrow geometry)
   - suburban (fewer neighbors, longer distances)
   - corridor (linear gNB layout, rapid boundary crossings)

3) **Propagation regimes**
   - mixed LoS/NLoS
   - heavy NLoS “tunnel” with deep loss and recovery
   - different interference scales (urban vs corridor)

4) **Behavioral corner cases** (ping‑pong traps)
   - intersections/turns
   - stop‑go movement
   - repeated in/out of coverage holes (tunnel oscillation)

The scenario catalog in [dataset.cpp](dataset.cpp) already covers these — you mainly need to **use all IDs/patterns** and **balance sampling**.

---

## 2) Scenario catalog (as implemented in dataset.cpp)

Each scenario defines:

- gNB geometry (positions + TX power)
- building layout (for LoS probability)
- pathloss model parameters by region
- interference scaling
- RLF/fallback thresholds (used by the internal HO/RLF logic in the generator)

### Scenario 1 — Dense Manhattan Grid

- ID: 1
- Area: 1000×1000 m
- gNBs: 9 (3×3 grid, ~333 m spacing)
- Buildings: dense grid blocks (high building density)
- Patterns:
  - 1A: pedestrian grid walk, **1.5 m/s**, 600 s
  - 1B: stop‑go waypoint walk (moving ~2 m/s with pauses), 600 s
  - 1C: fast vehicle loop, **15 m/s**, 600 s
- Why it matters: many strong neighbors → teaches hysteresis/TTT stability and anti ping‑pong.

### Scenario 2 — Urban Canyon

- ID: 2
- Area: long canyon (y: 0–2000 m, x: roughly -100..600 m)
- gNBs: 8 (two lines offset laterally)
- Buildings: two long “wall” buildings form canyon
- Patterns:
  - 2A: straight walk, **1.5 m/s**, 1333 s
  - 2B: fast walk with perpendicular turn segments, **~5 m/s**, 600 s
  - 2C: car moving fast, **20 m/s**, 100 s
- Why it matters: turn events + canyon geometry create rapid margin reversals.

### Scenario 3 — Suburban Residential

- ID: 3
- Area: 2000×2000 m
- gNBs: 12 (grid)
- Buildings: low density small buildings
- Patterns:
  - 3A: residential walk loop, **1.5 m/s**, 600 s
  - 3B: park walk sinusoid, **1.7 m/s**, 600 s
  - 3C: car commute with variable 20–30 m/s, 600 s
- Why it matters: fewer/softer neighbors and longer travel → teaches “don’t HO unless margin is real.”

### Scenario 4 — City Intersection

- ID: 4
- Area: x -150..450 m, y -200..450 m
- gNBs: 6 (around intersection + offsets)
- Buildings: 4 large blocks around intersection
- Patterns:
  - 4A: pedestrian diagonal crossing, **1.5 m/s**, 233 s
  - 4B: vehicle turning route, **15 m/s**, 600 s
  - 4C: stop‑go traffic cycles (move + long stop), 600 s
- Why it matters: classic ping‑pong trap (turning + blocks + stop‑go).

### Scenario 5 — High Speed Corridor

- ID: 5
- Area: 5000×400 m corridor
- gNBs: 7 (linear, ~700 m spacing)
- Buildings: sparse obstacles along corridor
- Patterns:
  - 5A: constant speed **30 m/s**, 600 s
  - 5B: variable speed rush hour **5→35→5 m/s** cycles, 600 s
  - 5C: emergency acceleration cycles (20/35/25 m/s segments), 600 s
- Why it matters: boundary crossings happen fast; if TTT is too low you get thrash, too high you get RLF.

### Scenario 6 — Mixed Urban

- ID: 6
- Area: 3000×2000 m
- gNBs: 21 (dense zone + transition + suburban)
- Buildings: density gradient (dense→transition→suburban)
- Patterns:
  - 6A: full commute with changing speeds **12/15/10/18/20 m/s**, 1200 s
  - 6B: downtown shopping small sinusoid, **2 m/s**, 600 s
  - 6C: multi‑mode day (walk→drive→walk→drive), **1.5/2/20 m/s**, 1200 s
- Why it matters: teaches context switching and prevents overfitting to one regime.

### Scenario 7 — NLoS Heavy Tunnel

- ID: 7
- Area: 1000×500 m with “tunnel strip” around y≈0
- gNBs: 6
- Buildings: two long blocks create NLoS tunnel
- Patterns:
  - 7A: walk through tunnel, **1.5 m/s**, 667 s
  - 7B: car through tunnel, **30 m/s**, 33 s
  - 7C: in‑and‑out oscillation (enter/stop/exit cycles), 600 s
- Why it matters: deep fades + recovery; forces the policy to learn RLF avoidance without ping‑pong.

---

## 3) Recommended training mix (the “perfect” scenario schedule)

### 3.1 Use ALL scenario×pattern combinations

Train on all 21 combinations (1A..7C). If you omit high‑speed patterns, you will almost certainly fail on vehicle tests.

### 3.2 Stratified sampling (avoid file‑bias)

Do **not** pick a random file uniformly; instead sample episodes by:

1) choose `scenario_id` uniformly from 1..7
2) choose `pattern` uniformly from {A,B,C}
3) choose a seed/run uniformly from your available list
4) choose a UE uniformly
5) choose a random start index in the UE trace (so episodes cover different parts of a long run)

This prevents scenario 6 (and walking speeds) from dominating training just because they have more files.

### 3.3 Curriculum (optional, but improves stability)

A practical curriculum that reduces early collapse:

- Stage 1 (stability): 1A, 3A, 4A (pedestrian) → learn “mostly stay”
- Stage 2 (turn/stop‑go): 1B, 2B, 4C → learn anti ping‑pong
- Stage 3 (vehicle): 1C, 3C, 4B → learn fast boundary handling
- Stage 4 (high speed corridor): 5A/5B/5C → learn TTT vs RLF tradeoff
- Stage 5 (mixed + tunnel): 6A/6C, 7A/7C → learn regime switching + deep fades

Once stable, switch to full stratified sampling over all 21 combos.

---

## 4) Train/validation/test split (avoid leakage)

Best practice: split by **seed**, not by rows.

Example (per scenario×pattern):

- Train seeds: 1–12
- Validation seeds: 13–15
- Test seeds: 16–20

If you want a harder generalization check:

- hold out one full scenario (e.g., scenario 7 tunnel) for testing
- or hold out all `pattern C` (fast) for testing

---

## 5) Concrete dataset generation recipe (ns‑3 runner)

The generator in [dataset.cpp](dataset.cpp) supports:

- `--scenarioId` (1..7)
- `--pattern` (A|B|C)
- `--duration` (seconds, 0 uses pattern default)
- `--ueCount`
- `--seed`
- `--tttMs` and `--hysDb` (for the generator’s internal HO logic)
- `--outputPrefix` (writes `_tick.csv`, `_events.csv`, `_summary.json`)

### Recommended run grid

For each scenarioId in 1..7 and pattern in A/B/C:

- seeds: 1..20
- ueCount: 20 (or 30 if you want larger datasets)
- duration: use pattern default (or cap long ones to 600–900s if storage is limited)

### Command template (example)

```bash
./ns3 run "... --scenarioId=6 --pattern=C --ueCount=20 --duration=0 --seed=17 --tttMs=200 --hysDb=4.0 --outputPrefix=results/train/s6_C_seed17"
```

Notes:

- The neighbor measurements are primarily determined by geometry + propagation and are largely independent of `tttMs/hysDb`.
- If storage allows, you can diversify generator `tttMs/hysDb` across a small set (e.g., (200,4.0), (250,4.5), (160,3.0)) to reduce any coupling to one HO baseline.

---

## 6) “Perfect training” sanity targets (what to monitor)

During dataset creation, check `_summary.json` and aggregate rates:

- handovers per minute: should be non‑zero across most scenario×pattern combos
- ping‑pong events: should exist in intersection/tunnel oscillation patterns, but not dominate
- RLF events: should exist in tunnel/corridor patterns (if RLF rate is ~0 across everything, something is wrong)

During PPO training, a healthy trajectory is:

- ping‑pong decreases over time without RLF increasing
- HO count decreases or stays near baseline while RLF decreases

---

## 7) If you only train on a subset (minimum viable set)

If compute/storage is limited, the smallest set that still covers the hard cases is:

- Urban grid: 1A + 1C
- Intersection: 4B + 4C
- Corridor: 5A + 5B
- Mixed: 6A
- Tunnel: 7C

But for best results, use all 21 combos.
