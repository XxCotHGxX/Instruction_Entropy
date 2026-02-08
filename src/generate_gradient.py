import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Load the subtask data
df = pd.read_csv('data/subtask_dataset.csv')

# 2. Cleanup and Log Transformation
# Use log1p to handle zeros and normalize the massive range of E
# E varies from 0 to 204,759. log(1+E) brings this into a 0-12 range.
df['log_e'] = np.log1p(df['e_sub'])
df['log_k'] = np.log1p(df['k_sub'])

# 3. Filter for 'Real' data points (exclude empty mappings)
df_plot = df[(df['e_sub'] > 0) & (df['k_sub'] > 0)].copy()

# 4. Generate the Heatmap using KDE (Kernel Density Estimate)
# This provides a smooth gradient instead of binary hexbins
import seaborn as sns
plt.figure(figsize=(12, 8))

# Create a smooth 2D density plot (KDE)
# This identifies the "Gravity Centers" of labor complexity
sns.kdeplot(
    data=df_plot, x="log_e", y="log_k", 
    fill=True, thresh=0, levels=100, cmap="viridis",
    cbar=True, cbar_kws={'label': 'Labor Concentration Density'}
)

# Overlay the actual points as small dots to show the raw data
plt.scatter(df_plot['log_e'], df_plot['log_k'], color='white', s=5, alpha=0.3, label='Requirement Observations')

# 5. Scientific Labels
plt.title('The Complexity Frontier: Labor Distribution Gradient', fontsize=16)
plt.xlabel('Instruction Entropy (Human Inference Density: log(1+E))', fontsize=12)
plt.ylabel('Artifact Coupling (Coordination Complexity: log(1+κ))', fontsize=12)

# Add the "Kink" coordinate indicator (Mean/Median intersection)
plt.axvline(df_plot['log_e'].mean(), color='red', linestyle='--', alpha=0.5)
plt.axhline(df_plot['log_k'].mean(), color='red', linestyle='--', alpha=0.5)
plt.text(df_plot['log_e'].mean()+0.2, df_plot['log_k'].mean()+0.2, 'The Kink (μ)', color='red', weight='bold')

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

os.makedirs('output', exist_ok=True)
plt.savefig('output/subtask_complexity_heatmap.png')
print("High-Fidelity Gradient Heatmap (KDE) generated successfully.")
