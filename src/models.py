import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample

def find_optimal_kink(df, target='ln_wage'):
    """Iteratively search for the log_entropy threshold that maximizes R-squared."""
    best_r2 = -1
    best_kink = 0
    
    # Test percentiles from 20% to 80% of log_entropy
    thresholds = np.percentile(df['log_entropy'], np.arange(20, 81, 5))
    
    for kink in thresholds:
        temp_df = df.copy()
        temp_df['log_entropy_high'] = np.maximum(0, temp_df['log_entropy'] - kink)
        try:
            model = smf.ols(f'{target} ~ log_entropy + log_entropy_high + ai_applicability_score', data=temp_df).fit()
            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_kink = kink
        except: continue
        
    return best_kink

def run_econometric_models():
    """Execute performance, wage elasticity, and production function regressions with log-log focus."""
    # Load Master Dataset
    df = pd.read_csv('data/master_dataset.csv')
    
    # Pre-processing
    df['log_entropy'] = np.log1p(df['instruction_entropy'])
    df['log_coupling'] = np.log1p(df['artifact_coupling'])
    
    # 1. Performance Model (Standardized OLS)
    print("--- Performance Model (Standardized) ---")
    df_std = df.copy()
    for col in ['log_entropy', 'log_coupling', 'ai_applicability_score']:
        df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
    
    model_p = smf.ols('success_label ~ log_entropy + log_coupling + ai_applicability_score', data=df_std).fit()
    print(model_p.summary())

    # 2. Wage Elasticity with Endogenous Kink (Log-Log)
    print("\n--- Wage Elasticity Model (Log-Log Piecewise) ---")
    df_wage = df.dropna(subset=['derived_wage'])
    df_wage = df_wage[df_wage['derived_wage'] > 0].copy()
    df_wage['ln_wage'] = np.log(df_wage['derived_wage'])
    
    kink_log_threshold = find_optimal_kink(df_wage)
    print(f"Optimal Complexity Kink identified at log(E) = {kink_log_threshold:.2f}")
    
    df_wage['log_entropy_high'] = np.maximum(0, df_wage['log_entropy'] - kink_log_threshold)
    model_w = smf.ols('ln_wage ~ log_entropy + log_entropy_high + ai_applicability_score', data=df_wage).fit()
    print(model_w.summary())

    # 3. Translog Production Function with Bootstrap
    print("\n--- Translog Production Function (Bootstrapped) ---")
    df_centered = df_wage.copy()
    for col in ['log_entropy', 'log_coupling']:
        df_centered[f'c_{col}'] = df_centered[col] - df_centered[col].mean()
    
    formula = 'ln_wage ~ c_log_entropy + c_log_coupling + I(0.5*c_log_entropy**2) + I(0.5*c_log_coupling**2) + I(c_log_entropy*c_log_coupling) + ai_applicability_score'
    
    n_iterations = 100
    bootstrap_params = []
    for i in range(n_iterations):
        sample = resample(df_centered)
        try:
            m = smf.ols(formula, data=sample).fit()
            bootstrap_params.append(m.params)
        except: continue
        
    boot_df = pd.DataFrame(bootstrap_params)
    print("Bootstrap Coefficients (Mean):")
    print(boot_df.mean())

    # Save results
    with open('output/refined_results.txt', 'w') as f:
        f.write(f"Refined Results (Log-Kink Threshold: {kink_log_threshold:.2f})\n")
        f.write("========================================\n\n")
        f.write("Piecewise Log-Log Coefficients:\n")
        f.write(model_w.params.to_string())

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df_wage['log_entropy'], df_wage['ln_wage'], alpha=0.6, label='Observations')
    
    # Plot piecewise fit
    log_e_axis = np.linspace(df_wage['log_entropy'].min(), df_wage['log_entropy'].max(), 100)
    log_e_high = np.maximum(0, log_e_axis - kink_log_threshold)
    pred_df = pd.DataFrame({'log_entropy': log_e_axis, 'log_entropy_high': log_e_high, 'ai_applicability_score': df_wage['ai_applicability_score'].mean()})
    plt.plot(log_e_axis, model_w.predict(pred_df), color='red', label='Piecewise Fit')
    
    plt.axvline(kink_log_threshold, color='black', linestyle='--', label=f'Kink (log E={kink_log_threshold:.2f})')
    plt.title('Log-Log Complexity Kink Detection')
    plt.xlabel('log(Instruction Entropy)')
    plt.ylabel('ln(Wage)')
    plt.legend()
    plt.savefig('output/complexity_kink_refined.png')

    # --- NEW: Density Heatmap (Hexbin) showing "The Cliff" ---
    plt.figure(figsize=(12, 8))
    
    # We want to see the density of successes vs failures
    # Hexbin of log_e and log_k weighted by success
    hb = plt.hexbin(df_wage['log_entropy'], df_wage['log_coupling'], 
                    C=df_wage['success_label'], gridsize=20, 
                    cmap='RdYlGn', mincnt=1)
    
    cb = plt.colorbar(hb)
    cb.set_label('AI Success Probability (Empirical)')
    
    plt.title('The Complexity Kink: AI Success Phase Transition')
    plt.xlabel('Instruction Entropy (log E)')
    plt.ylabel('Artifact Coupling (log κ)')
    
    # Add a marker for the "Expert Zone"
    plt.text(df_wage['log_entropy'].max()*0.7, df_wage['log_coupling'].max()*0.8, 
             'THE EXPERT ZONE\n(AI Failure Cliff)', 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12, color='red')
             
    plt.savefig('output/complexity_kink_heatmap.png')
    print("New Density Heatmap (Hexbin) generated in output/complexity_kink_heatmap.png")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    run_econometric_models()
