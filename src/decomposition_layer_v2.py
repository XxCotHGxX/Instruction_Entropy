import os
import pandas as pd
import numpy as np
import zlib
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def get_mdl_size(text):
    """Calculates the Minimum Description Length (MDL) using zlib compression."""
    if not text: return 0
    return len(zlib.compress(text.encode('utf-8')))

def extract_requirements(brief_text):
    """Decomposes a brief into semantically discrete requirements."""
    pattern = r'(?:^|\n)(?:\s*[-*]|\s*\d+\.)\s+(.*)'
    requirements = re.findall(pattern, brief_text)
    return [r.strip() for r in requirements if len(r.strip()) > 10]

def map_req_to_files(req, deliverable_dir):
    """Maps requirements to solution assets via technical keyword overlap."""
    if not os.path.exists(deliverable_dir):
        return []
    keywords = set(re.findall(rf'\b([a-zA-Z]{{4,}})\b', req.lower()))
    stop_words = {'this', 'that', 'with', 'from', 'your', 'will', 'into', 'proper', 'format', 'using', 'needed'}
    keywords = keywords - stop_words
    
    relevant_files = []
    for root, dirs, files in os.walk(deliverable_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(5000).lower()
                    if any(k in content for k in keywords):
                        relevant_files.append(file_path)
            except: pass
    return list(set(relevant_files))

def calculate_hardened_metrics(req, files):
    """
    Calculates E and Kappa using unit-less Information Theory measures.
    E (Inference Density) = MDL(Solution) / MDL(Instruction)
    Kappa (Coordination Complexity) = Normalized Reference Density
    """
    if not files:
        return 0, 0
    
    # 1. Inference Density (E)
    mdl_instruction = get_mdl_size(req)
    
    solution_text = ""
    logic_exts = ('.tex', '.py', '.js', '.html', '.css', '.md', '.txt', '.c', '.h', '.mat')
    for f in files:
        if f.lower().endswith(logic_exts):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fo:
                    solution_text += fo.read(20000) + " "
            except: pass
            
    mdl_solution = get_mdl_size(solution_text)
    
    # E is the 'Expansion Ratio' of Information
    e_hardened = mdl_solution / mdl_instruction if mdl_instruction > 0 else 0
    
    # 2. Coordination Complexity (Kappa)
    # Using Unique Symbol Density as a proxy for state-dependency (NMI proxy)
    symbols = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{5,}\b', solution_text)
    unique_symbols = len(set(symbols))
    
    # Normalize by the log-volume of the solution to get a density metric
    kappa_hardened = (unique_symbols / np.log(len(solution_text) + 1)) if len(solution_text) > 0 else 0
    
    return round(e_hardened, 4), round(kappa_hardened, 4)

def decompose_projects():
    """Builds the expanded dataset using the hardened methodology."""
    rli_base = 'data/rli_public_set'
    subtask_data = []
    
    if not os.path.exists('data/master_dataset.csv'):
        print("Master dataset missing.")
        return pd.DataFrame()

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
            e, k = calculate_hardened_metrics(req, relevant_files)
            
            subtask_data.append({
                'project_id': task_id,
                'requirement': req[:100],
                'e_hardened': e,
                'k_hardened': k,
                'success': row['success_label'],
                'ln_wage_eq': np.log(row['equilibrium_wage']) if row['equilibrium_wage'] > 0 else 0,
                'automation_exposure': row['ai_applicability_score']
            })
            
    df = pd.DataFrame(subtask_data)
    df.to_csv('data/subtask_dataset_v2.csv', index=False)
    print(f"Decomposition complete. N={len(df)} requirements processed.")
    return df

if __name__ == "__main__":
    decompose_projects()
