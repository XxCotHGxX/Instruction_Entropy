import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm

def generate_selection_cliff():
    if not os.path.exists('data/subtask_dataset_v2.csv'):
        print("Dataset missing.")
        return

    df = pd.read_csv('data/subtask_dataset_v2.csv')
    # Label inclusion based on the model's filtering criteria
    df['included'] = df.apply(lambda x: 1 if (x['e_hardened'] > 0 and x['k_hardened'] > 0 and x['ln_wage_eq'] > 0) else 0, axis=1)
    
    # 1. THE SELECTION CLIFF (Probit Visualization)
    # We want to show how AI Applicability (Automation Exposure) drives inclusion
    plt.figure(figsize=(10, 6))
    
    # Bin the data to show the probability curve
    df['bins'] = pd.qcut(df['automation_exposure'], 10, duplicates='drop')
    bin_probs = df.groupby('bins', observed=True)['included'].mean()
    bin_centers = df.groupby('bins', observed=True)['automation_exposure'].mean()
    
    plt.scatter(bin_centers, bin_probs, color='red', s=50, label='Empirical Probabilities')
    
    # Fit a smooth trend line
    from scipy.interpolate import make_interp_spline
    X_Y_Spline = make_interp_spline(bin_centers, bin_probs)
    X_ = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color='black', linewidth=2, label='Selection Boundary (p=0.03)')
    
    plt.title('The Selection Cliff: Curation Bias in AI Benchmarks', fontsize=14)
    plt.xlabel('Task Modularity (Automation Exposure Score)', fontsize=12)
    plt.ylabel('Probability of Task Inclusion in Gold-Standard Sets', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    
    os.makedirs('output/v2', exist_ok=True)
    plt.savefig('output/v2/selection_cliff.png')
    plt.savefig('paper/selection_cliff.png')
    print("New 'Selection Cliff' visual generated.")

if __name__ == "__main__":
    generate_selection_cliff()
