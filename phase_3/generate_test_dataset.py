import os
import glob
import random
import numpy as np
import pandas as pd

def generate_noise_dataset(source_dir, dest_dir, num_files=20, noise_std=2.0):
    """
    Randomly selects CSVs from Phase 1, injects procedural Gaussian RF noise, and saves them
    into a synthetic unseen `test_dataset` for Phase 3 RL validation.
    """
    os.makedirs(dest_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(source_dir, "*_tick.csv"))
    
    if not all_files:
        raise RuntimeError(f"No source files found in {source_dir}")
        
    np.random.seed(42)
    random.seed(42)  # Fixed seed for pure reproducibility of the test dataset
    
    # Shuffle and pick files (take 20)
    selected_files = random.sample(all_files, min(num_files, len(all_files)))
    
    print(f"Generating synthetic unseen datasets in {dest_dir}...")
    
    for count, file_path in enumerate(selected_files):
        df = pd.read_csv(file_path)
        base_name = os.path.basename(file_path)
        new_name = base_name.replace('_tick', '_test_noise')
        dest_path = os.path.join(dest_dir, new_name)
        
        # Define target columns to perturb
        rsrp_cols = ['serving_rsrp_dbm'] + [f'n{i}_rsrp_dbm' for i in range(1, 7)]
        sinr_cols = ['serving_sinr_db']
        
        for col in rsrp_cols + sinr_cols:
            if col in df.columns:
                # Add gaussian noise (e.g. fast fading synthetic interference)
                noise = np.random.normal(0, noise_std, size=len(df))
                
                # Apply the noise only where the value isn't purely a NaN or zero
                mask = df[col].notna() & (df[col] != 0.0)
                df.loc[mask, col] = df.loc[mask, col] + noise[mask]
                
        df.to_csv(dest_path, index=False)
        print(f"[{count+1}/{len(selected_files)}] Generated {new_name}")
        
    print(f"\nSuccessfully generated {len(selected_files)} synthetic unseen datasets with ±{noise_std}dB RF noise!")

if __name__ == "__main__":
    src = r"E:\5g_handover\dataset_phase1"
    dst = r"E:\5g_handover\phase_3\test_dataset"
    generate_noise_dataset(src, dst)
