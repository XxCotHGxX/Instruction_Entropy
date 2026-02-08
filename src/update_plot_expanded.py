import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

def find_optimal_kink(df, target='ln_wage_eq'):
    best_r2 = -1
    best_kink = 0
    thresholds = np.percentile(df['log_e'], np.arange(20, 81, 5))
    
    for kink in thresholds:
        temp_df = df.copy()
        temp_df['log_e_high'] = np.maximum(0, temp_df['log_e'] - kink)
        try:
            model = smf.ols(f'{target} ~ log_e + log_e_high + ai_score', data=temp_df).fit()
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_kink = kink
        except: continue
    return best_kink

def run():
    df = pd.read_csv('data/subtask_dataset.csv')
    df = df[df['e_sub'] > 0].copy()
    df['log_e'] = np.log(df['e_sub'])
    
    kink_threshold = find_optimal_kink(df)
    df['log_e_high'] = np.maximum(0, df['log_e'] - kink_threshold)
    
    model = smf.ols('ln_wage_eq ~ log_e + log_e_high + ai_score', data=df).fit()
    print(model.summary())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['log_e'], df['ln_wage_eq'], alpha=0.4, color='gray', label=f'Sub-tasks (N={len(df)})')
    
    log_e_axis = np.linspace(df['log_e'].min(), df['log_e'].max(), 100)
    log_e_high = np.maximum(0, log_e_axis - kink_threshold)
    pred_df = pd.DataFrame({'log_e': log_e_axis, 'log_e_high': log_e_high, 'ai_score': df['ai_score'].mean()})
    plt.plot(log_e_axis, model.predict(pred_df), color='red', linewidth=2, label='Piecewise RKD Fit')
    
    plt.axvline(kink_threshold, color='black', linestyle='--', label=f'Kink Threshold (log E={kink_threshold:.2f})')
    plt.title('Regression Kink Design: The Non-Linear Labor Floor (Expanded N=156)', fontsize=14)
    plt.xlabel('log(Instruction Entropy)', fontsize=12)
    plt.ylabel('ln(Market Equilibrium Wage)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/complexity_kink_expanded.png')
    # Also update the paper copy
    os.makedirs('paper', exist_ok=True)
    plt.savefig('paper/complexity_kink_expanded.png')
    print("Expanded Kink Plot generated: output/complexity_kink_expanded.png")

if __name__ == "__main__":
    run()
