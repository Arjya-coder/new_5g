# Novelty Statement (Paper Structure)

## A. Core Scientific Contributions

1. We propose a hierarchical handover control framework where a deterministic safety-aware handover engine is combined with a learning-based parameter optimizer.

2. We design a dual-mode baseline handover policy that separates emergency handover behavior (MUST_HO) from proactive optimization behavior, improving safety and control interpretability.

3. We introduce temporal stability gating for candidate eligibility by combining TTT hold logic with strongest-neighbor persistence across consecutive ticks, which reduces unstable switching.

4. We add predictive neighbor ranking based on short-horizon margin forecasting (slope-informed margin), instead of relying only on instantaneous signal comparison.

5. We keep decisions explainable by exposing structured diagnostics (reason, risk context, zone, candidate list), which supports analysis and debugging in networking experiments.

## B. Algorithm-1 Specific Novelties

6. Cross-signal risk-aware decision logic is used in baseline control by jointly considering filtered RSRP, filtered SINR, CQI, distance zone, and mobility context.

7. The handover trigger design combines emergency conditions and proactive conditions with soft confidence penalties under ping-pong risk, rather than a single threshold trigger.

8. The candidate processing pipeline explicitly separates: distance filtering, event holding, eligibility, predictive ranking, and decision selection, making the baseline modular and reproducible.

## C. Algorithm-2 (PPO) Specific Novelties

9. We use parameter-space RL control (TTT/HYS adaptation) instead of direct target-cell selection, preserving telecom safety structure while still enabling online optimization.

10. We define a compact 23-dimensional state representation that mixes radio quality, topology margin, temporal trend, mobility, and recent event-rate context for stable policy learning.

11. We use a structured 15-action lattice that jointly adjusts TTT step and HYS offset, enabling smooth but expressive policy updates.

12. We introduce velocity-aware RLF penalty shaping in the reward, so failure penalties increase with speed, matching higher real-world risk at high mobility.

13. We optimize a multi-objective reward balancing RLF events, ping-pong events, handover cost, SINR behavior, parameter smoothness, no-op bias, and handover frequency.

14. We train with control-interval interaction (step_n) and variable-step discounting/GAE, aligning learning updates with practical control cadence instead of per-tick overreaction.

15. We integrate scenario-aware RLF event modeling and explicit ping-pong event detection rules in the training environment to create realistic and measurable supervision signals.

16. We include handover event de-duplication (in_handover guard) so one physical handover is not repeatedly counted as multiple events.

## D. Practical and Reproducibility Contributions

17. We use bounded, normalized state construction with explicit validity checks (dimension and range assertions), improving training robustness and reproducibility.

18. We provide a full train-evaluate pipeline with periodic checkpointing, metric logging, and parameter-history export, supporting transparent comparison and ablation studies.

## E. Claim-Safe Notes for Paper Writing

19. In this implementation branch, the Phase-2 embedded baseline uses A3-based holding logic (not full A3+A5), so this variant difference should be clearly stated.

20. In this implementation branch, ping-pong and handover-failure events are primarily penalized and logged, not always used as hard episode termination conditions.

## Suggested One-Paragraph Contribution Summary

This work presents a safety-aware hierarchical handover framework that combines deterministic radio control logic with PPO-based online parameter adaptation. The baseline controller contributes dual-mode emergency/proactive triggering, temporal eligibility stabilization, and predictive margin ranking, while remaining interpretable through explicit diagnostics. The PPO layer contributes compact cross-layer state design, structured TTT/HYS action control, and a velocity-aware RLF penalty that improves risk alignment under high mobility. Together with control-interval training, scenario-aware event modeling, and reproducible logging/evaluation, the framework provides a practical and research-ready approach for robust 5G handover optimization.
