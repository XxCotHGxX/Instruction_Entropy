import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample
from scipy.stats import norm
import os

def run_hardened_analysis():
    # Load the new V2 dataset
    df = pd.read_csv('data/subtask_dataset_v2.csv')
    
    # Filter for valid observations
    df_clean = df[(df['e_hardened'] > 0) & (df['k_hardened'] > 0) & (df['ln_wage_eq'] > 0)].copy()
    
    print(f"Dataset Hardening: Filtered {len(df)} requirements down to N={len(df_clean)} valid subtasks.")
    
    # 1. HECKMAN FIRST STAGE: Selection into inclusion
    df['included'] = df.apply(lambda x: 1 if (x['e_hardened'] > 0 and x['k_hardened'] > 0 and x['ln_wage_eq'] > 0) else 0, axis=1)
    
    print("\n--- HECKMAN FIRST STAGE (PROBIT) ---")
    first_stage = sm.Probit(df['included'], sm.add_constant(df['automation_exposure'])).fit()
    print(first_stage.summary())
    
    # Calculate Inverse Mills Ratio (IMR)
    # λ(x) = φ(xβ) / Φ(xβ)
    xb = first_stage.predict(sm.add_constant(df_clean['automation_exposure']), transform=False)
    imr = norm.pdf(xb) / norm.cdf(xb)
    df_clean['IMR'] = imr
    
    # 2. INFERENCE PERFORMANCE MODEL (LOGIT)
    print("\n--- INFERENCE PERFORMANCE MODEL (LOGIT) ---")
    logit_formula = 'success ~ e_hardened + k_hardened'
    try:
        model_logit = smf.logit(logit_formula, data=df_clean).fit()
        print(model_logit.summary())
    except:
        print("Logit Model failed to converge (Perfect Separation likely).")
        model_logit = None
    
    # 3. MARKET VALUATION MODEL (MEAN-CENTERED TRANSLOG)
    df_clean['c_log_e'] = np.log(df_clean['e_hardened']) - np.log(df_clean['e_hardened']).mean()
    df_clean['c_log_k'] = np.log(df_clean['k_hardened']) - np.log(df_clean['k_hardened']).mean()
    
    print("\n--- MARKET VALUATION MODEL (BOOTSTRAP) ---")
    translog_formula = 'ln_wage_eq ~ c_log_e + c_log_k + I(0.5*c_log_e**2) + I(0.5*c_log_k**2) + I(c_log_e*c_log_k) + IMR'
    
    n_iterations = 200
    bootstrap_results = []
    projects = df_clean['project_id'].unique()
    
    for i in range(n_iterations):
        sample_projects = resample(projects)
        sample_df = pd.concat([df_clean[df_clean['project_id'] == p] for p in sample_projects])
        try:
            m = smf.ols(translog_formula, data=sample_df).fit()
            bootstrap_results.append(m.params)
        except: continue
        
    boot_df = pd.DataFrame(bootstrap_results)
    
    # Print results without special characters
    print("\nBootstrapped Coefficients (Mean):")
    print(boot_df.mean())
    print("\nBootstrapped Significance (p-value proxy):")
    for col in boot_df.columns:
        mean_val = boot_df[col].mean()
        if mean_val > 0:
            p_val = (boot_df[col] < 0).mean() * 2
        else:
            p_val = (boot_df[col] > 0).mean() * 2
        print(f"{col}: p ~= {min(1.0, p_val):.4f}")

    # Save outputs
    os.makedirs('output/v2', exist_ok=True)
    with open('output/v2/hardened_results.txt', 'w') as f:
        f.write("HARDENED SCIENTIFIC RESULTS (V2)\n")
        f.write("================================\n\n")
        if model_logit:
            f.write("LOGIT SUCCESS MODEL:\n")
            f.write(model_logit.summary().as_text())
        f.write("\n\nTRANSLOG WAGE MODEL (BOOTSTRAP):\n")
        f.write(boot_df.mean().to_string())
        f.write("\n\nSIGNIFICANCE:\n")
        for col in boot_df.columns:
            mean_val = boot_df[col].mean()
            p_val = (boot_df[col] < 0).mean() * 2 if mean_val > 0 else (boot_df[col] > 0).mean() * 2
            f.write(f"{col}: p ~= {min(1.0, p_val):.4f}\n")

if __name__ == "__main__":
    run_hardened_analysis()
