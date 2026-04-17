PHASE 1: BASELINE ALGORITHM 1 TESTING
├── Run ns-3 simulations online with FIXED TTT/HYS
├── Collect decisions + outcomes
├── Analyze: "How many HOs? RLF events? Ping-pong?"
└── Output: Baseline performance metrics

PHASE 2: RL TRAINING
├── Load online ns3 datasets
├── Train PPO to learn optimal TTT/HYS
├── For each scenario: "What parameters work best?"
└── Output: Learned policy (TTT/HYS adjustments)

PHASE 3: RL DEPLOYMENT TESTING
├── Run ns-3 with RL-tuned TTT/HYS
├── Collect same metrics as Phase 1
├── Compare: "RL vs Baseline"
└── Output: Improvements (fewer HOs, fewer RLF, better QoE)