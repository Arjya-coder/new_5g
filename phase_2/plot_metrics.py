import os
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(log_dir="logs", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_files = glob.glob(os.path.join(log_dir, "metrics_*.json"))
    if not metrics_files:
        print("No metrics files found.")
        return
        
    latest_metrics = max(metrics_files, key=os.path.getctime)
    
    with open(latest_metrics, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 1. Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['total_reward'], label="Total Episode Reward", color='b', alpha=0.5)
    plt.plot(df['episode'], df['total_reward'].rolling(window=20).mean(), label="Reward Moving Average (20)", color='r')
    plt.title("Learning Curve: Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()
    
    # 2. Tradeoff Curve
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('RLF Count', color=color)
    ax1.plot(df['episode'], df['rlf_count'].rolling(window=20).mean(), color=color, label="RLFs (Moving Avg)")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Ping-Pong Count', color=color)  
    ax2.plot(df['episode'], df['pp_count'].rolling(window=20).mean(), color=color, label="Ping-Pongs (Moving Avg)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    plt.title("Mitigation Tradeoff: RLF vs Ping-Pongs")
    plt.savefig(os.path.join(save_dir, 'tradeoff_curve.png'))
    plt.close()
    
    # 3. Bar Chart of Parameter Distribution from analysis
    analysis_files = glob.glob(os.path.join(log_dir, "param_analysis.json"))
    if analysis_files:
        with open(analysis_files[0], 'r') as f:
            analysis = json.load(f)
            
        zones = list(analysis.keys())
        avg_ttt = [analysis[z]['avg_ttt'] for z in zones]
        avg_hys = [analysis[z]['avg_hys'] for z in zones]
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.bar(zones, avg_ttt, color='orange', alpha=0.6, label='Avg TTT (ms)')
        ax1.set_ylabel('Time-To-Trigger (ms)')
        for i, v in enumerate(avg_ttt):
            ax1.text(i, v + 5, f"{v:.0f}", ha='center')
            
        ax2 = ax1.twinx()
        ax2.plot(zones, avg_hys, color='red', marker='o', linewidth=2, label='Avg HYS (dB)')
        ax2.set_ylabel('Hysteresis (dB)')
        for i, v in enumerate(avg_hys):
            ax2.text(i, v + 0.1, f"{v:.1f}", ha='center')
            
        plt.title("Parameter Distributions by Geographical Zone")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'action_distribution.png'))
        plt.close()

if __name__ == "__main__":
    plot_metrics()
