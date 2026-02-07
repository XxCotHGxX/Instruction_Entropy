import os
import pandas as pd
import numpy as np
import tiktoken
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if not text: return 0
    return len(tokenizer.encode(text))

def extract_requirements(brief_text):
    pattern = r'(?:^|\n)(?:\s*[-*]|\s*\d+\.)\s+(.*)'
    requirements = re.findall(pattern, brief_text)
    requirements = [r.strip() for r in requirements if len(r.strip()) > 10]
    return requirements

def map_req_to_files(req, deliverable_dir):
    if not os.path.exists(deliverable_dir):
        return []
    keywords = set(re.findall(rf'\b([a-zA-Z]{{4,}})\b', req.lower()))
    stop_words = {'this', 'that', 'with', 'from', 'your', 'will', 'into', 'proper', 'format', 'using', 'needed'}
    keywords = keywords - stop_words
    
    relevant_files = []
    for root, dirs, files in os.walk(deliverable_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if any(k in file.lower() for k in keywords):
                relevant_files.append(file_path)
                continue
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000).lower()
                    if any(k in content for k in keywords):
                        relevant_files.append(file_path)
            except: pass
    return list(set(relevant_files))

def calculate_subtask_metrics(req, files):
    """Calculates E and Kappa for a specific subset of the project."""
    if not files:
        return 0, 0
    
    # MDL-based Entropy (E): The ratio of 'Information Out' to 'Information In'
    req_tokens = count_tokens(req)
    deliverable_text = ""
    for f in files:
        # Filter for text-based solution assets only to prevent binary noise
        if f.endswith(('.tex', '.py', '.js', '.html', '.css', '.md', '.txt', '.c', '.h', '.mat')):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    deliverable_text += file_obj.read(50000) + " "
            except: pass
    
    sol_tokens = count_tokens(deliverable_text)
    # E is the 'Inference Density'
    e_sub = sol_tokens / req_tokens if req_tokens > 0 else 0
    
    # State Dependency Density (Kappa): Cross-asset symbol mapping
    fan_out = len(files)
    max_depth = 0
    if len(files) > 1:
        try:
            base_dir = os.path.commonpath(files)
            for f in files:
                max_depth = max(max_depth, f.count(os.sep) - base_dir.count(os.sep))
        except: pass
            
    kappa_sub = (fan_out * 0.4) + (max_depth * 0.6)
    return e_sub, kappa_sub

def decompose_projects():
    rli_base = 'data/rli_public_set'
    subtask_data = []
    orig_df = pd.read_csv('data/master_dataset.csv')
    
    for _, row in orig_df.iterrows():
        task_id = row['Task ID']
        print(f"Decomposing task: {task_id}")
        folder_path = os.path.join(rli_base, task_id)
        brief_path = os.path.join(folder_path, 'project', 'brief.md')
        deliverable_dir = os.path.join(folder_path, 'human_deliverable')
        if not os.path.exists(brief_path): continue
        with open(brief_path, 'r', encoding='utf-8') as f:
            brief_text = f.read()
        requirements = extract_requirements(brief_text)
        for req in requirements:
            relevant_files = map_req_to_files(req, deliverable_dir)
            e_sub, k_sub = calculate_subtask_metrics(req, relevant_files)
            subtask_data.append({
                'project_id': task_id,
                'requirement': req[:100] + '...',
                'e_sub': e_sub,
                'k_sub': k_sub,
                'success': row['success_label'],
                'ln_wage_eq': np.log(row['equilibrium_wage']) if row['equilibrium_wage'] > 0 else 0,
                'ai_score': row['ai_applicability_score']
            })
    sub_df = pd.DataFrame(subtask_data)
    sub_df.to_csv('data/subtask_dataset.csv', index=False)
    print(f"Decomposition complete: Created {len(sub_df)} sub-tasks from 10 projects.")
    return sub_df

def run_clustered_regression(df):
    # Filter for valid ranges
    df = df[(df['e_sub'] > 0) & (df['k_sub'] > 0) & (df['ln_wage_eq'] > 0)].copy()
    df['log_e'] = np.log(df['e_sub'])
    df['log_k'] = np.log(df['k_sub'])
    
    formula = 'ln_wage_eq ~ log_e + log_k + I(0.5*log_e**2) + I(0.5*log_k**2) + I(log_e*log_k) + ai_score'
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['project_id']})
    print("\n--- Clustered Hedonic Translog Regression (Market Equilibrium) ---")
    print(model.summary())
    
    with open('output/decomposition_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    # --- NEW: Density Heatmap (Hexbin) showing "The Frontier" ---
    plt.figure(figsize=(12, 8))
    # Filter for reasonable E range for visualization and ensure we have enough points
    df_plot = df[df['e_sub'] < 5000].copy()
    
    # Increase gridsize for more resolution and change to a log scale for density
    # to show the subtle variations in concentration
    hb = plt.hexbin(df_plot['log_e'], df_plot['log_k'], 
                    gridsize=40, 
                    cmap='Spectral_r', 
                    bins='log', 
                    mincnt=1)
    
    cb = plt.colorbar(hb)
    cb.set_label('Concentration of Tasks (Log Scale)')
    
    plt.title('The Complexity Frontier: Distribution of Professional Labor', fontsize=16)
    plt.xlabel('Instruction Entropy (Requirements Inference Required)', fontsize=12)
    plt.ylabel('Artifact Coupling (Coordination Across Assets)', fontsize=12)
    
    # Add a note about the kink
    plt.annotate('Tipping Point (The Kink)', xy=(df_plot['log_e'].median(), df_plot['log_k'].median()),
                 xytext=(df_plot['log_e'].median()+1, df_plot['log_k'].median()+1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.savefig('output/subtask_complexity_heatmap.png')
    print("New High-Resolution Density Map (Log-Scaled) generated.")

if __name__ == "__main__":
    print("Starting decomposition...")
    df = decompose_projects()
    if len(df) > 0:
        run_clustered_regression(df)
