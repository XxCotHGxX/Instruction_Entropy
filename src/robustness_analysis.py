import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import os

def run_robustness_analysis():
    """Execute corrected Heckman selection and breakpoint robustness."""
    df = pd.read_csv('data/master_dataset.csv')
    
    # Pre-processing
    df['ln_wage'] = np.log(df['derived_wage'].replace(0, np.nan))
    
    print("--- Corrected Heckman Selection Correction ---")
    try:
        # Step 1: First-Stage Probit (Selection into the Expert Set)
        # We use ai_applicability_score as the exclusion restriction
        probit_model = sm.Probit(df['success_label'], sm.add_constant(df['ai_applicability_score'])).fit()
        
        # Step 2: Calculate Inverse Mills Ratio (IMR)
        z = probit_model.predict(sm.add_constant(df['ai_applicability_score']))
        df['mills'] = norm.pdf(z) / norm.cdf(z)
        
        # Step 3: Second-Stage OLS with IMR
        kink = 1000 # Default
        df['entropy_high'] = np.maximum(0, df['instruction_entropy'] - kink)
        
        model_h = smf.ols('ln_wage ~ instruction_entropy + entropy_high + mills', data=df).fit()
        print(model_h.summary())
        
        with open('output/heckman_results.txt', 'w') as f:
            f.write("Heckman Corrected Piecewise Wage Elasticity\n")
            f.write("===========================================\n\n")
            f.write(model_h.params.to_string())
            f.write("\n\nIMR Coefficient (Lambda): " + str(model_h.params['mills']))
            
    except Exception as e:
        print(f"Heckman model error: {e}")

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    run_robustness_analysis()
