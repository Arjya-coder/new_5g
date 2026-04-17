import matplotlib.pyplot as plt
import numpy as np
import os

results = {
    'Baseline': {'RLFs': 82, 'PingPongs': 122},
    'PPO_Agent': {'RLFs': 48, 'PingPongs': 2239}
}

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(2)
width = 0.35

rects1 = ax.bar(x - width/2, [results['Baseline']['RLFs'], results['PPO_Agent']['RLFs']], width, label='Radio Link Failures (RLF)', color='darkred')
rects2 = ax.bar(x + width/2, [results['Baseline']['PingPongs'], results['PPO_Agent']['PingPongs']], width, label='Ping Pongs (A->B->A)', color='darkorange')

ax.set_ylabel('Total Count across all Unseen Data')
ax.set_title('Phase 3 Deployment Validation: Baseline vs RL\n(Note: PPO PingPongs spiked due to evaluation sampling vs greedy)')
ax.set_xticks(x)
ax.set_xticklabels(['Phase 1 Static Baseline', 'Phase 2 Intelligent PPO'])
ax.legend()

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/comparison_results.png')
print("Successfully generated comparison plots.")
