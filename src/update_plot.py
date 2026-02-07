import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('data/subtask_dataset.csv')
df = df[(df['e_sub'] > 0) & (df['k_sub'] > 0)]
df['log_e'] = np.log1p(df['e_sub'])
df['log_k'] = np.log1p(df['k_sub'])

plt.figure(figsize=(12, 8))
hb = plt.hexbin(df['log_e'], df['log_k'], C=df['success'], gridsize=15, cmap='RdYlGn', mincnt=1)
cb = plt.colorbar(hb)
cb.set_label('AI Success Rate (Green = High, Red = Low)')

plt.title('The Complexity Kink: Mapping the AI Productivity Cliff', fontsize=16)
plt.xlabel('Instruction Entropy (Human Inference Required)', fontsize=12)
plt.ylabel('Artifact Coupling (Coordination Across Many Files)', fontsize=12)

# Annotate
# (Removed annotation as per user request)

os.makedirs('output', exist_ok=True)
plt.savefig('output/subtask_complexity_heatmap.png')
print("Heatmap updated with LinkedIn-friendly labels.")
