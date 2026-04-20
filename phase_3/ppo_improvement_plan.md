# Improving the Phase‑2 PPO Handover Agent vs Baseline (Phase‑3 Findings)

Date: 2026‑04‑19

This note explains what `comparison_results_v3.png` is telling us and gives a **prioritized, concrete roadmap** to make the PPO agent outperform the baseline in Phase‑3 evaluation.

---

## 1) What the Phase‑3 results actually say (symptoms)

Phase‑3 evaluation compared:

- **Baseline** (static parameters): `TTT=250 ms`, `HYS=4.5 dB`
- **Trained PPO** (Phase‑2 model): actor/critic loaded from `phase_2/models/final`

**Aggregate counts (3 files):**

| Metric | Baseline | PPO | Relative change (PPO vs Baseline) |
|---|---:|---:|---:|
| Handovers | 2,597 | 15,589 | **+500%** (worse) |
| RLF | 1,229 | 1,339 | **+9%** (worse) |
| Ping‑Pong | 73 | 2,026 | **+2,675%** (worse) |

**Interpretation:** the PPO policy is **far too aggressive**. It is triggering far more handovers than the baseline, and many of those are **rapid A→B→A oscillations** (ping‑pong). RLF is not improving, so the extra handovers are not “buying” reliability.

That failure pattern almost always comes from one of these (often multiple at once):

1) **Reward misalignment**: reward unintentionally makes “handover early/often” profitable.
2) **State/data contract problems**: the agent is trained on (or deployed with) distorted state features.
3) **Control‑rate problem**: you let the agent change parameters too frequently, creating oscillation.
4) **Dataset sampling bias / distribution shift**: training over‑represents some scenarios.

---

## 2) Before changing reward/dataset: validate the metric plumbing

If the measurement is off, you can chase the wrong fix.

### 2.1 Handover counting consistency

In Phase‑2 training, `TrainingEnv` intentionally avoids double‑counting handovers using an `in_handover` gate.

In Phase‑3 evaluation (`phase_3/compare_models.py`), the HO counter currently treats `decision['action']>0` as a handover immediately.

**Action:** make sure Phase‑3 counts HO the same way training counts HO:

- Prefer `decision['ho_occurred']` (if available) over `action!=0`.
- Add an `in_handover` latch (same logic as `TrainingEnv`).

Even if this reduces absolute HO counts, the **ping‑pong explosion** strongly suggests PPO is still unstable, but it’s worth fixing so the metrics reflect reality.

### 2.2 Ping‑pong definition

Training and Phase‑3 both use a strict rule: “A→B then back to A within 1 second”. Keep that definition identical across training/eval.

---

## 3) The highest‑impact root cause (confirmed): training environment hides RLF dynamics

In Phase‑2, the offline environment in `phase_2/train_rl.py` creates serving SINR like this when the serving cell differs from the CSV row’s `serving_cell`:

- it uses a heuristic `serving_sinr = max(-20.0, row['serving_sinr_db'] - 5.0)`

Then Phase‑2 RLF detection uses:

- `if sinr_current < -20: rlf_counter += 1`

**Result:** the environment *almost never produces* `sinr < -20`, therefore the agent almost never experiences RLF events during training.

This shows up clearly in the training metrics (`phase_2/logs/metrics_20260416_105239.json`):

- mean `rlf_count` ≈ **0.13** per episode
- many late episodes have `rlf_count = 0`

But Phase‑3 evaluation shows RLF is a major phenomenon (baseline RLF=1229). That’s a massive mismatch.

### 3.1 Fix the offline environment (must do)

In `OfflineNs3Env._process_current_row()` (Phase‑2 training), reconstruct serving measurements from the **neighbor columns** matching the **currently connected** cell:

- If `algo1.serving_cell_id` is the current serving cell, find its index in `n{i}_id`.
- Set:
  - `serving_rsrp = n{i}_rsrp_dbm`
  - `serving_sinr = n{i}_sinr_db`
  - `distance_serving = n{i}_d_m`

If that cell ID is not present in neighbors, fall back to the row’s `serving_*` fields (but do **not** clamp SINR).

This change:

- makes training dynamics match Phase‑3 deployment assumptions
- allows SINR to legitimately drop below −20, so RLF is learnable
- fixes the “distance_serving is always 200m” problem (your Phase‑1 CSVs do not include `serving_d_m`)

**If you do only one thing: do this first.** Until the env is correct, reward tuning is mostly noise.

---

## 4) Reward: why the current PPO can rationally learn “handover too much”

The current reward in `phase_2/algo_2.py::RLModule.compute_reward()` is:

- base: `+0.1` every step
- HO penalty: `-0.1` when HO occurs
- ping‑pong penalty: `-0.8`
- RLF penalty: `-1.2`
- HO frequency penalty: `-0.05 * recent_ho_count`
- smoothness penalty: `-0.1*(|ΔTTT| + |ΔHYS|)`
- SINR shaping: `0.1*sinr_delta` (clipped) + small absolute SINR term
- then `clip(r, -1, +1)`

### 4.1 Key problem: HO can be “free”

Because you always add `+0.1`, a single HO penalty `-0.1` is often canceled out immediately, and the agent can still net positive reward via SINR terms.

If ping‑pong detection is strict and RLF events are rare (see Section 3), the policy can learn:

> “make TTT small, switch often, chase momentary SINR improvements”

Which matches the Phase‑3 symptom: massive HO + ping‑pong.

### 4.2 A better reward structure (recommended)

Use an explicit **cost** and minimize it. A practical reward template:

\[
 r_t = w_{\Delta} \cdot \Delta\text{SINR}_t + w_{\text{abs}} \cdot f(\text{SINR}_t)
 - w_{\text{HO}}\,\mathbb{1}[\text{HO}]
 - w_{\text{PP}}\,\mathbb{1}[\text{PP}]
 - w_{\text{RLF}}\,\mathbb{1}[\text{RLF}]
 - w_{\text{rate}}\cdot \text{HO\_rate}_t
 - w_{\text{osc}}\cdot \|\Delta\theta_t\|
\]

Where \(\theta\) are your parameters (TTT/HYS).

Concrete changes to try first:

1) **Remove** the unconditional `+0.1` per step (or reduce it to ~`+0.01`).
2) Increase HO cost: start with `w_HO = 0.3` to `0.6`.
3) Increase HO‑rate cost: e.g., `w_rate = 0.1` and define `HO_rate` as #HO in last 1–2 seconds.
4) Add “unproductive HO” penalty:
   - if HO occurs and SINR does not improve after 0.5–1.0 s, add extra penalty.
5) Avoid hard reward clipping early (or clip after scaling). Clipping at ±1 can destroy gradient signal.

**Goal:** PPO should only do more HOs when it measurably reduces RLF risk without creating oscillation.

---

## 5) Action space & control rate: 100 ms updates encourage oscillation

Right now, the agent can change TTT/HYS every 0.1 s. Even with smoothness penalties, that’s a lot of control authority.

Recommended guardrails:

- **Update only every 0.5–1.0 s** (hold the chosen parameters for multiple ticks).
- **Minimum dwell time** after HO: do not allow HO triggers for 1.0 s (or enforce higher HYS/TTT during dwell).
- Add a deployment “safety layer”: if ping‑pong rate exceeds a threshold in a sliding window, temporarily revert to baseline parameters.

These are not “cheating” — real mobility management systems do exactly this.

---

## 6) Dataset bias: do you need to change the dataset?

### 6.1 What we measured

- Phase‑3 test set contains only scenario IDs {1,4,7} and speeds {1.5, 15}.
- Phase‑1 training set (sample) is dominated by scenario 6 and has broader speed coverage.

This means **sampling bias** is real: if you sample uniformly by file/row, scenario 6 can dominate training, and Phase‑3 doesn’t include it.

### 6.2 What to change (recommended)

You don’t necessarily need a brand new dataset format, but you should change **how you sample** and ensure the dataset supports the state variables.

**A) Fix missing/constant features**

- `serving_d_m` is not in Phase‑1 CSVs, so distance is effectively constant unless reconstructed from neighbor distances.
- `cqi_serving` is constant in training/eval (hard‑coded 10). Either compute CQI from SINR or include it in dataset.

**B) Balance sampling**

Instead of choosing random file + random UE (which implicitly weights by file length and scenario prevalence), sample episodes **uniformly by scenario**:

- sample a `scenario_id` uniformly
- then choose a file/UE within that scenario

**C) Add “do nothing is correct” segments**

If most of the dataset is near cell edges or fast mobility, the agent learns that “handover is always helpful.” Ensure training includes:

- long stable center‑cell segments where HO should be rare
- medium mobility where baseline works well

---

## 7) Model selection & validation: don’t deploy “final” blindly

Training shows high variance (HO counts range 0 to ~58 per episode; reward swings negative/positive). The final checkpoint is not necessarily best.

Recommended:

- keep a **validation set** (hold out scenario IDs or entire files)
- every N episodes, evaluate greedy policy on validation
- save and deploy the **best checkpoint by validation cost**, not the last episode

Define a single validation score (example):

\[
J = 1.0\cdot\text{RLF} + 0.2\cdot\text{PingPong} + 0.05\cdot\text{HO}
\]

Tune weights to match your priorities.

---

## 8) A practical step‑by‑step plan (what to do next)

### Step 1 — Fix the Phase‑2 offline environment (highest ROI)

- reconstruct serving metrics from neighbor columns
- remove SINR clamping
- reconstruct serving distance from neighbor distances

Retrain PPO and re‑run Phase‑3 comparison.

### Step 2 — Strengthen anti‑ping‑pong and HO cost in reward

- remove `+0.1` per step
- increase HO + HO‑rate penalties
- add “unproductive HO” penalty

Retrain; compare again.

### Step 3 — Reduce control rate / add dwell time guardrails

- only update parameters every 0.5–1.0 s
- enforce dwell time after HO

Retrain/evaluate.

### Step 4 — Fix dataset sampling bias

- sample uniformly by scenario_id (and optionally speed bins)

Retrain/evaluate.

---

## 9) What “success” should look like

For Phase‑3 test traces, a realistic first target is:

- **Ping‑pong**: within 2× baseline (i.e., ≤ ~150) 
- **Handovers**: within 1.2× baseline (i.e., ≤ ~3100)
- **RLF**: ≤ baseline (i.e., ≤ ~1229)

Once stable, try to push RLF down without blowing up HO/PP.

---

## 10) If you want, I can implement these fixes

The first concrete code changes I’d implement are:

1) Update `phase_2/train_rl.py` offline env reconstruction (serving RSRP/SINR/distance).
2) Update `phase_3/compare_models.py` HO counting to match `TrainingEnv`.

After that, we can iterate on reward weights with a small experiment grid.
