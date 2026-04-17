import os
import glob
import random
import pandas as pd
from algo_2 import Algorithm1, RLModule, PPOAgent, TrainingEnv, train_ppo, analyze_parameter_patterns

class OfflineNs3Env:
    """
    Offline wrapper simulating the ns-3 C++ sandbox using pre-generated datasets.
    """
    def __init__(self, dataset_dir):
        self.csv_files = glob.glob(os.path.join(dataset_dir, "*_tick.csv"))
        if not self.csv_files:
            raise RuntimeError(f"No *_tick.csv files found in {dataset_dir}")
        self.current_df = None
        self.row_idx = 0
        self.current_ue = None
        
    def reset(self, algo1):
        file_path = random.choice(self.csv_files)
        self.current_df = pd.read_csv(file_path)
        ues = self.current_df['ue_id'].unique()
        self.current_ue = random.choice(ues)
        self.current_df = self.current_df[self.current_df['ue_id'] == self.current_ue].reset_index(drop=True)
        self.row_idx = 0
        
        return self._process_current_row(algo1, 160, 3.0)
        
    def step(self, ttt_eff, hys_eff, algo1):
        if self.is_done():
            # Failsafe bounds check to hold at max index
            self.row_idx = len(self.current_df) - 1
        output = self._process_current_row(algo1, ttt_eff, hys_eff)
        self.row_idx += 1
        return output
        
    def is_done(self):
        return self.current_df is None or self.row_idx >= len(self.current_df) - 1
        
    def _process_current_row(self, algo1, ttt_eff, hys_eff):
        row = self.current_df.iloc[self.row_idx]
        
        rsrp_neighbors = []
        neighbor_ids = []
        distance_neighbors = []
        
        for n_i in range(1, 7):
            n_id = row.get(f'n{n_i}_id', -1)
            if pd.isna(n_id) or n_id == -1: continue
            neighbor_ids.append(int(n_id))
            rsrp_neighbors.append(float(row[f'n{n_i}_rsrp_dbm']))
            distance_neighbors.append(float(row[f'n{n_i}_d_m']))
            
        if algo1.serving_cell_id == 0:
            algo1.serving_cell_id = int(row['serving_cell'])
            
        serving_rsrp = -140.0
        serving_sinr = -20.0
        
        if int(row['serving_cell']) == algo1.serving_cell_id:
            serving_rsrp = float(row['serving_rsrp_dbm'])
            serving_sinr = float(row['serving_sinr_db'])
        else:
            found = False
            for i, nid in enumerate(neighbor_ids):
                if nid == algo1.serving_cell_id:
                    serving_rsrp = rsrp_neighbors[i]
                    # Simulate relative SINR degradation when disconnected from primary physics
                    serving_sinr = max(-20.0, float(row['serving_sinr_db']) - 5.0) 
                    found = True
                    break
            if not found:
                serving_rsrp, serving_sinr = -140.0, -20.0
        
        # Determine serving distance reliably
        distance_serving = float(row.get('serving_d_m', 200.0))
                
        decision = algo1.step(
            rsrp_serving=serving_rsrp,
            sinr_serving=serving_sinr,
            cqi_serving=10, 
            distance_serving=distance_serving, 
            rsrp_neighbors=rsrp_neighbors,
            neighbor_ids=neighbor_ids,
            distance_neighbors=distance_neighbors,
            velocity=float(row.get('speed_mps', 15.0)),
            now_s=float(row['time_s']),
            TTT_eff=ttt_eff,
            HYS_eff=hys_eff
        )
        return decision

if __name__ == "__main__":
    dataset_dir = r"E:\5g_handover\dataset_phase1"
    ns3_env = OfflineNs3Env(dataset_dir)
    algo1 = Algorithm1()
    rl_module = RLModule()
    
    training_env = TrainingEnv(ns3_env, algo1, rl_module)
    agent = PPOAgent(state_dim=20, action_dim=15)
    
    train_ppo(agent, rl_module, training_env, num_episodes=350, rollout_horizon=512, save_dir="models", log_dir="logs")
    
    # After training, parse the params
    import glob
    latest_params = max(glob.glob('logs/param_history_*.json'), key=os.path.getctime)
    analyze_parameter_patterns(latest_params, 'logs/param_analysis.json')
