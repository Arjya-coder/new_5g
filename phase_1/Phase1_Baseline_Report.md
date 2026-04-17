# Phase 1: Baseline 5G Handover Algorithm Validation Report

## 1. Executive Summary
Phase 1 of the 5G Handover Pipeline project established a rigorous baseline using a static, traditional A3 handover logic (**Algorithm 1**). The pipeline executed synthetic signal generation using an `ns-3` environment wrapped in a custom Python orchestration script, generating over 2.2 GB of raw UE telemetry across 7 distinct topographical scenarios. The baseline results clearly highlight the rigid limitations of static `TTT` (Time-to-Trigger) and `HYS` (Hysteresis) constraints when exposed to dynamic obstacles, tunnel fading, and aggressive ping-pong zones.

## 2. Testing Methodology
* **Environment:** `ns-3` (C++) inside Windows Subsystem for Linux (WSL).
* **Pipeline Logic:** Python `run_pipeline.py` processing dataset CSV traces synchronously.
* **Algorithm 1:** Ground truth deterministic handover model.
* **Volume:** 70 Total Runs (7 Scenarios × 2 Patterns × 5 Random Seeds).
* **Static Constraints Assessed:** `TTT = 160ms`, `HYS = 3dB`.

## 3. Aggregated Performance Data

| Scenario | Topography Description | Pattern | Avg Handovers | Avg RLFs 💥 | Avg Ping-Pongs 🏓 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **S1** | **Dense Manhattan Grid** | **A** (Pedestrian) | 7,341 | 90 | 6,909 |
| **S1** | Dense Manhattan Grid | **B** (Random Waypt) | 8,188 | 95 | 7,857 |
| **S2** | **Suburban (Trees/NLOS)** | **A** (Looping Drive) | 1,033 | 32 | 801 |
| **S2** | Suburban (Trees/NLOS) | **B** (Stop & Go) | 35,768 | 906 | 35,502 |
| **S3** | **Open Park (Strong LOS)** | **A** (Perimeter Walk) | 90 | 47 | 37 |
| **S3** | Open Park (Strong LOS) | **B** (Random Walk) | 1,591 | 53 | 1,379 |
| **S4** | **Rural Highway LOS** | **A** (Fast Drive) | 21,640 | **0** | 21,227 |
| **S4** | Rural Highway LOS | **B** (Random Speeds) | 12,409 | **0** | 12,045 |
| **S5** | **Mixed Highway NLOS** | **A** (Linear Drive) | 48 | 23 | **2** |
| **S5** | Mixed Highway NLOS | **B** (Variable Traffic) | 68 | 36 | **1** |
| **S6** | **Downtown Commute** | **A** (Walk/Drive Mix) | 7,625 | 204 | 7,383 |
| **S6** | Downtown Commute | **B** (Heavy Traffic) | 30,467 | 1,103 | 30,157 |
| **S7** | **Tunnel Stress Test** | **A** (Fast Tunneling)| 12,749 | 1,578 | 12,532 |
| **S7** | Tunnel Stress Test | **B** (Jam In Tunnel) | 11,944 | 1,586 | 11,304 |

---

## 4. Analytical Observations

### 4.1 The Ping-Pong Crisis (LOS Overlap)
In purely unobstructed, Line-of-Sight corridors like **Scenario 4 (Rural Highway)**, the UE encounters vast cell-edge overlap areas where multiple gNBs exhibit equivalent high RSRP values. 
* Algorithm 1 blindly oscillates connections between cells at the slightest micro-fluctuation of interference, resulting in an apocalyptic **~21,000 Ping-Pong** handovers. 
* The zero-RLF rate proves coverage is absolutely perfect—but the volatile ping-pong rate would obliterate network throughput.
* **Diagnosis:** A static `HYS = 3dB` is far too weak in overlapping LOS zones. It must dynamically expand to resist bouncing.

### 4.2 The Tunnel Radio Link Failure (RLF) Breakdown
In **Scenario 7 (Tunnel Stress Test)**, UE trajectories immediately slam into physical blockage, experiencing near-instantaneous signal degradation.
* The standard static `TTT = 160ms` requires the UE to "wait and verify" the neighboring cell for a long period before handing over.
* By the time the TTT timer expires, the serving signal has entirely plunged beneath the `-122 dBm` crash threshold.
* This results in **over 1,500 system crashes (RLF sequences)** per environment run. 
* **Diagnosis:** The `TTT` parameter must be radically shortened to near `0ms` (instantaneous reactionary handover) the moment an aggressive NLOS object (tunnel) is mathematically detected.

### 4.3 Static Resilience
In controlled, sparsely structured zones like **Scenario 5**, the default constants performed extremely well (`Avg 1 Ping-Pong`), proving that static constants work if the environment coincidentally matches their mathematical assumptions. 

## 5. Conclusion & Action Items for Phase 2
Phase 1 systematically proves that a static A3 condition cannot optimally survive modern dynamic network topographies. Algorithm 1 establishes a comprehensive baseline of failure points.

To proceed, **Phase 2 (RL Training)** will bypass static rigidity.
* We will construct **Algorithm 2** utilizing a Proximal Policy Optimization (PPO) reinforcement learning model.
* The RL agent will utilize the `dataset_phase1` metrics as a replay-buffer environment.
* The mission is to train the Agent to actively augment its `TTT` and `HYS` parameters in real-time by interpreting the velocity, LOS probability, and RSRP deterioration velocity of the UE.
