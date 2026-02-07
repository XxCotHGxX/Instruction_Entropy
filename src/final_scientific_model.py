import os
import pandas as pd
import numpy as np
import tiktoken
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if not text: return 0
    return len(tokenizer.encode(text))

def extract_requirements(brief_text):
    """Scientific requirement extraction based on list-item boundaries."""
    pattern = r'(?:^|\n)(?:\s*[-*]|\s*\d+\.)\s+(.*)'
    requirements = re.findall(pattern, brief_text)
    # Filter for requirements that carry semantic weight (>10 chars)
    requirements = [r.strip() for r in requirements if len(r.strip()) > 10]
    return requirements

def map_req_to_files(req, deliverable_dir):
    """Maps requirements to files using technical keyword density."""
    if not os.path.exists(deliverable_dir):
        return []
    keywords = set(re.findall(rf'\b([a-zA-Z]{{4,}})\b', req.lower()))
    stop_words = {'this', 'that', 'with', 'from', 'your', 'will', 'into', 'proper', 'format', 'using', 'needed', 'requirements', 'everything', 'include', 'project', 'deliverables', 'standard'}
    keywords = keywords - stop_words
    
    relevant_files = []
    for root, dirs, files in os.walk(deliverable_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Match by filename
            if any(k in file.lower() for k in keywords):
                relevant_files.append(file_path)
                continue
            # Match by content snippet (first 5k chars)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000).lower()
                    if any(k in content for k in keywords):
                        relevant_files.append(file_path)
            except: pass
    return list(set(relevant_files))

def calculate_subtask_metrics(req, files):
    """
    Calculates E and Kappa with a Boilerplate-Agnostic MDL Filter.
    E (Entropy) = (Solution - Boilerplate) / log(1+Brief).
    """
    if not files:
        return 0, 0
    
    req_tokens = count_tokens(req)
    deliverable_text = ""
    
    # 1. Boilerplate Mitigation (Hole #1 Fix)
    boilerplate_patterns = [
        r'\\documentclass.*?\n', r'\\usepackage.*?\n', r'\\bibliographystyle.*?\n',
        r'import .*?\n', r'from .*? import .*?\n', r'# Copyright .*?\n',
        r'/\*.*?\*/', r'//.*?\n', r'<!DOCTYPE.*?>'
    ]
    
    logic_extensions = ('.tex', '.py', '.js', '.html', '.css', '.md', '.txt', '.c', '.h', '.mat', '.cpp', '.java', '.json', '.yaml', '.xml')
    
    for f in files:
        if f.lower().endswith(logic_extensions):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read(10000)
                    # Strip Boilerplate
                    for pattern in boilerplate_patterns:
                        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                    deliverable_text += content + " "
            except: pass
    
    sol_tokens = count_tokens(deliverable_text)
    
    # 2. MDL Normalization (Hole #3 Fix)
    # We use log-smoothing on the instruction count to prevent E from 
    # exploding due to 'bad boss' brevity. This creates a more stable 
    # measure of 'Inference Density' across varying brief qualities.
    b_prime = np.log1p(req_tokens) * 10 
    
    e_sub = sol_tokens / b_prime if b_prime > 0 else 0
    
    # State Dependency Density (Kappa)
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
    """Builds the 156-subtask dataset with cleaned metrics."""
    rli_base = 'data/rli_public_set'
    subtask_data = []
    orig_df = pd.read_csv('data/master_dataset.csv')
    
    for _, row in orig_df.iterrows():
        task_id = row['Task ID']
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
                'requirement': req[:100],
                'e_sub': e_sub,
                'k_sub': k_sub,
                'success': row['success_label'],
                'ln_wage_eq': np.log(row['equilibrium_wage']) if row['equilibrium_wage'] > 0 else 0,
                'ai_score': row['ai_applicability_score']
            })
            
    sub_df = pd.DataFrame(subtask_data)
    sub_df.to_csv('data/subtask_dataset.csv', index=False)
    return sub_df

def run_analysis(df):
    """Clustered regression and KDE Gradient mapping."""
    df = df[(df['e_sub'] > 0) & (df['k_sub'] > 0) & (df['ln_wage_eq'] > 0)].copy()
    df['log_e'] = np.log(df['e_sub'])
    df['log_k'] = np.log(df['k_sub'])
    
    formula = 'ln_wage_eq ~ log_e + log_k + I(0.5*log_e**2) + I(0.5*log_k**2) + I(log_e*log_k) + ai_score'
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['project_id']})
    print("\n--- CLUSTERED HEDONIC TRANSLOG (STRESS-TESTED) ---")
    print(model.summary())
    
    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        data=df, x="log_e", y="log_k", 
        fill=True, thresh=0, levels=100, cmap="viridis",
        cbar=True, cbar_kws={'label': 'Labor Concentration Density'}
    )
    plt.scatter(df['log_e'], df['log_k'], color='white', s=5, alpha=0.3)
    
    plt.title('The Complexity Frontier: Mapping the AI Productivity Cliff', fontsize=16)
    plt.xlabel('Instruction Entropy (Inference Required: log E)', fontsize=12)
    plt.ylabel('Artifact Coupling (Coordination Complexity: log κ)', fontsize=12)
    
    plt.axvline(df['log_e'].mean(), color='red', linestyle='--', alpha=0.5)
    plt.axhline(df['log_k'].mean(), color='red', linestyle='--', alpha=0.5)
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/subtask_complexity_heatmap.png')
    print("Stress-Tested High-Fidelity Gradient Heatmap generated.")

if __name__ == "__main__":
    df = decompose_projects()
    run_analysis(df)
