import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.formula.api as smf

def find_optimal_kink(df, target='ln_wage_eq'):
    best_r2 = -1
    best_kink = 0
    # Use log_e from the hardened metrics
    thresholds = np.percentile(df['log_e'], np.arange(20, 81, 5))
    for kink in thresholds:
        temp_df = df.copy()
        temp_df['log_e_high'] = np.maximum(0, temp_df['log_e'] - kink)
        try:
            model = smf.ols(f'{target} ~ log_e + log_e_high', data=temp_df).fit()
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_kink = kink
        except: continue
    return best_kink

def generate_hardened_visuals():
    if not os.path.exists('data/subtask_dataset_v2.csv'):
        print("Hardened dataset missing.")
        return

    df = pd.read_csv('data/subtask_dataset_v2.csv')
    # Use the same filtering as the model
    df = df[(df['e_hardened'] > 0) & (df['k_hardened'] > 0) & (df['ln_wage_eq'] > 0)].copy()
    
    # Information Theory logs
    df['log_e'] = np.log(df['e_hardened'])
    df['log_k'] = np.log(df['k_hardened'])
    
    # 1. THE TECHNOLOGICAL FRONTIER (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        data=df, x="log_e", y="log_k", 
        fill=True, thresh=0, levels=100, cmap="magma",
        cbar=True, cbar_kws={'label': 'Labor Concentration Density'}
    )
    plt.scatter(df['log_e'], df['log_k'], color='white', s=10, alpha=0.5)
    
    plt.title('The Technological Frontier: Mapping the High-Entropy Regime (N=57)', fontsize=16)
    plt.xlabel('Inference Density (MDL Ratio: log E)', fontsize=12)
    plt.ylabel('Coordination Complexity (Reference Density: log κ)', fontsize=12)
    
    plt.axvline(df['log_e'].mean(), color='cyan', linestyle='--', alpha=0.5, label='Sample Mean E')
    plt.axhline(df['log_k'].mean(), color='cyan', linestyle='--', alpha=0.5, label='Sample Mean κ')
    
    os.makedirs('output/v2', exist_ok=True)
    plt.savefig('output/v2/subtask_complexity_heatmap.png')
    plt.savefig('paper/subtask_complexity_heatmap.png')
    print("Hardened Frontier Map generated.")

    # 2. THE STRUCTURAL BREAK (RKD Plot)
    kink_threshold = find_optimal_kink(df)
    df['log_e_high'] = np.maximum(0, df['log_e'] - kink_threshold)
    
    model = smf.ols('ln_wage_eq ~ log_e + log_e_high', data=df).fit()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['log_e'], df['ln_wage_eq'], alpha=0.5, color='gray', label=f'Professional Sub-tasks (N={len(df)})')
    
    log_e_axis = np.linspace(df['log_e'].min(), df['log_e'].max(), 100)
    log_e_high = np.maximum(0, log_e_axis - kink_threshold)
    pred_df = pd.DataFrame({'log_e': log_e_axis, 'log_e_high': log_e_high})
    plt.plot(log_e_axis, model.predict(pred_df), color='red', linewidth=2, label='Piecewise Structural Break (RKD)')
    
    plt.axvline(kink_threshold, color='black', linestyle='--', label=f'Breakpoint (log E={kink_threshold:.2f})')
    plt.title('Structural Break in the AI Production Function (MDL-Based)', fontsize=14)
    plt.xlabel('log(Inference Density)', fontsize=12)
    plt.ylabel('ln(Market Equilibrium Wage)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig('output/v2/complexity_kink_expanded.png')
    plt.savefig('paper/complexity_kink_expanded.png')
    print("Hardened Structural Break Plot generated.")

if __name__ == "__main__":
    generate_hardened_visuals()
