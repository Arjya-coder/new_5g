# PPO Agent Improvement Report (Baseline → Better-than-Baseline)

Date: 2026‑04‑19

Scope: Based on Phase‑3 evaluation plot `comparison_results_v3.png` and the current training/evaluation code in this workspace.

---

## 1) What the Phase‑3 result means (plain English)

From Phase‑3 evaluation outputs:

- Baseline (static): **HOs=2,597**, **RLFs=1,229**, **Ping‑Pongs=73**
- PPO (trained): **HOs=15,589**, **RLFs=1,339**, **Ping‑Pongs=2,026**

This is not a “small tuning” issue — it’s a **policy stability / objective mismatch** issue:

- PPO is **triggering far too many handovers**.
- A lot of those are **A→B→A oscillations** (ping‑pong explosion).
- RLF doesn’t improve; therefore extra handovers aren’t buying robustness.

When you see **HO and ping‑pong blow up together**, the usual root causes are:

1) training environment/state mismatch (policy learns on a distorted world)
2) reward makes HO “cheap” compared to SINR shaping
3) parameter control is too fast (policy can thrash TTT/HYS each 0.1s)
4) partial observability (policy does not know its current TTT/HYS)

---

## 2) Checking the “Three Real Problems” claim against the actual data/code

You proposed:

1) training dataset speed bias (most critical)
2) reward makes handovers free
3) delta action space causes oscillation

### 2.1 Speed bias: partially true, but not the main smoking gun

**Phase‑3 test dataset (all rows):**

- speeds: only **1.5 m/s (60k rows)** and **15 m/s (30k rows)**
- mean speed: **6.0 m/s**
- scenario_ids: **{1, 4, 7}**, exactly 30k rows each

So Phase‑3 is a **very narrow** distribution (two speeds, three scenarios).

**Phase‑1 training dataset (sampled from `dataset_phase1.zip`):**

- speeds include: **1.5, 2, 5, 10, 12, 15, 18, 30, 35 m/s** (plus rare 0)
- mean speed (sample): **~11.53 m/s**
- scenario_ids in sample: **6 dominates**, then 7, 4, 5, 1, 2

So: it is **not true** that training data is “always 1.5 m/s”.

Why it can *look* that way in your logs anyway:

- `phase_2/logs/param_history_*.json` stores only the **last episode’s** per-step parameters (because `TrainingEnv.reset()` clears `param_history`). In the current logs, those last episodes happened to be low-speed (1.5–2.0 m/s), which can mislead a “training speed” summary.
- `OfflineNs3Env.reset()` samples **uniformly by file**, not by scenario/speed. In `dataset_phase1.zip`, most tick files are low-speed (25/70 at 1.5 m/s, plus 5 at 1.7 and 10 at 2.0), so naive file-uniform sampling is biased toward walking speeds.

What *is* true:

- training data is **scenario‑imbalanced** (scenario 6 dominates), and Phase‑3 tests only {1,4,7}
- the Phase‑3 speed distribution is **narrower** than training, but this alone doesn’t explain the PPO instability

**Conclusion:** dataset imbalance is real, but **not the most critical root cause** of the Phase‑3 collapse.

### 2.2 “RLF never happens in training”: true in practice, but due to the training environment logic

The dataset itself contains RLF labels:

- Phase‑3 `rlf_event` rate: **0.00493** (0.493%), 444 events / 90k rows
- Phase‑1 zip sample `rlf_event` rate: **0.00603** (0.603%), 8,677 events / 1.44M rows

So you do **not** need to “inject 0.3% RLF events” into the dataset — they already exist.

But in PPO training logs (`phase_2/logs/metrics_20260416_105239.json`):

- mean `rlf_count` per 512‑step episode ≈ **0.131**

That’s extremely low, given the dataset’s RLF label rate.

**Why:** the Phase‑2 offline environment (`phase_2/train_rl.py::OfflineNs3Env`) reconstructs serving SINR in a way that can suppress deep fades after HO, and it also feeds a mostly constant serving distance (see Section 3). This makes the agent’s experience of “RLF risk” far weaker than what Phase‑3 exposes.

**Conclusion:** you’re right that the agent didn’t learn RLF avoidance well — but it’s mostly a **world‑model/state issue**, not a lack of RLF examples in the dataset.

### 2.3 Reward makes HO effectively cheap: yes (high confidence)

In `phase_2/algo_2.py::RLModule.compute_reward()`:

- there is a constant **`+0.1`** per step
- HO penalty is **`-0.1`**

So in many states the base reward can cancel HO cost, and SINR shaping can dominate. This can rationally produce:

> “Switch early/often to chase short‑term SINR improvements.”

That matches Phase‑3 behavior (HOs + ping‑pong explode).

**Conclusion:** this is a real issue and should be changed.

### 2.4 Delta action space causes oscillation: yes, and there’s a deeper issue (partial observability)

Actions are deltas: ΔTTT step, ΔHYS.

But the **state vector does not include current TTT or current HYS**.

The actor is a feed‑forward MLP (no memory), so it cannot infer the hidden “current parameter” from history.

This turns your problem into a POMDP where the policy is effectively open‑loop with respect to its own control state.

**Conclusion:** either

- switch to an **absolute action space** (choose TTT/HYS directly), or
- keep delta actions but **add current TTT/HYS to the state**

This is a top‑priority fix.

---

## 3) The biggest confirmed root cause: broken/constant state features during training

### 3.1 Serving distance is effectively constant in training

Phase‑2 offline env uses:

- `distance_serving = row.get('serving_d_m', 200.0)`

But the CSV schema in your datasets does **not** include `serving_d_m` (serving distance is represented via neighbor `n{i}_d_m` fields).

So training distance becomes **always 200 m**, which forces:

- zone classification = **CELL_CENTER** most of the time
- rlf_risk’s distance term rarely activates

This is directly visible in `phase_2/logs/param_analysis.json`:

- only one zone appears: `CELL_CENTER`
- `avg_distance` = **200.0**

If the agent never experiences HANDOFF/CRITICAL/EDGE zones during training, it cannot learn stable anti‑ping‑pong behavior for boundary conditions.

### 3.2 Serving SINR reconstruction mismatch between training and Phase‑3 evaluation

Phase‑3 evaluation tries to reconstruct serving measurements from neighbor columns for the *connected* cell.

Phase‑2 training env uses a heuristic when serving cell differs from the row’s serving cell:

- it may clamp/alter SINR (`max(-20, ...)`) instead of using true neighbor SINR

This can reduce the frequency/severity of SINR drops after HO and therefore weaken RLF learning.

---

## 4) What to do (in order) — practical, code‑level steps

### Step 1 (must‑do): Fix the offline environment state reconstruction

File: `phase_2/train_rl.py` (`OfflineNs3Env._process_current_row`)

**Goal:** when the currently connected cell is `algo1.serving_cell_id`, read that cell’s RSRP/SINR/distance from the neighbor columns.

Implementation rule:

- If `algo1.serving_cell_id == row['serving_cell']`, you can use serving_* columns.
- Else, find the index `i` such that `n{i}_id == algo1.serving_cell_id`.
  - use `n{i}_rsrp_dbm`, `n{i}_sinr_db`, and `n{i}_d_m` as serving metrics.
- If not found, fall back to row’s serving metrics (but do not clamp SINR).

This one fix typically:

- restores realistic zone distribution
- restores realistic RLF pressure
- makes training dynamics match Phase‑3 evaluation assumptions

### Step 2: Fix partial observability (state must include current TTT/HYS)

File: `phase_2/algo_2.py`

Add 2 dimensions:

- current `ttt_eff` (normalized)
- current `hys_eff` (normalized)

Where to get them:

- `TrainingEnv.step()` already knows `ttt_eff`/`hys_eff` and can pass them into `build_state_vector`.
- Phase‑3 evaluator also knows the currently applied parameters and should pass them the same way.

If you keep the state at 20‑dim, you can drop 2 weaker features; otherwise move to 22‑dim.

### Step 3: Reward redesign (make stability cheap, oscillation expensive)

File: `phase_2/algo_2.py::RLModule.compute_reward`

Changes to try first:

1) Remove constant `+0.1` (or reduce to +0.01).
2) Increase HO penalty (start with −0.3 to −0.6).
3) Add a *dwell‑scaled* HO penalty:
   - e.g., extra penalty if HO occurs when `time_since_last_ho < 1.0s`
4) Add “unproductive HO” penalty:
   - if HO occurs and SINR does not improve within 0.5–1.0s, penalize.
5) Avoid aggressive clipping at ±1 early; clipping can kill learning signal.

### Step 4: Action space & control‑rate stabilization

You have two good options:

**Option A (recommended): absolute action space**

- choose TTT from `[100,160,200,240,320]`
- choose HYS from a small grid (e.g., 1.0 to 6.0 in 0.5 steps)

This prevents runaway drift.

**Option B: keep delta actions but limit update rate**

- only apply a new action every 0.5–1.0 seconds
- otherwise hold parameters constant

Both reduce oscillation.

### Step 5: Fix dataset sampling bias (speed + scenario) before heavy reward tuning

The dataset format is fine; what needs to change is sampling.

In `OfflineNs3Env.reset()`:

- sample **uniformly by scenario_id** and **speed bin**
- then pick file + UE within that scenario

This prevents scenario 6 from dominating training if your deployment/test cares about {1,4,7}.

If your *actual* training directory truly contains only walking-speed traces, then this step becomes “regenerate dataset with mixed speeds” instead of “resample.”

---

## 5) Minimal experiment plan (so you don’t waste weeks)

Run these in order; each should be evaluated via the existing Phase‑3 comparison script.

1) **Env fix only** (Step 1). Retrain short (e.g., 50–100 episodes) just to see if HO/PP collapse improves.
2) **Env fix + state fix** (add current TTT/HYS). Retrain.
3) **Reward fix** (remove +0.1; stronger HO penalties). Retrain.
4) **Control‑rate guardrail** (hold actions 0.5–1.0s). Retrain.
5) **Scenario‑balanced sampling**. Retrain.

Stop as soon as you reach:

- ping‑pong within ~2× baseline
- handovers within ~1.2× baseline
- RLF ≤ baseline

Then tune further for RLF improvements.

---

## 6) My bottom-line “thought” on your diagnosis

- ✅ Reward issue: **agree** (needs change).
- ✅ Delta/oscillation issue: **agree**, and the bigger problem is **state missing current TTT/HYS**.
- ⚠️ Speed‑bias issue: **imbalance exists**, but “training is always 1.5 m/s” is **not supported** by the Phase‑1 dataset; the more urgent problem is the **offline environment/state reconstruction** (distance constant, serving SINR mismatch) which prevents the agent from learning the right tradeoffs.

If you want, I can implement Step 1 + Step 2 in code and set up a quick retrain + Phase‑3 re‑evaluation loop.
