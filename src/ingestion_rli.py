import os
import json
import pandas as pd
from huggingface_hub import snapshot_download

def download_rli_data():
    """Download the RLI Public Set from Hugging Face."""
    print("Downloading RLI Public Set from Hugging Face...")
    try:
        repo_id = 'cais/rli-public-set'
        # Local dir to store the dataset
        local_dir = 'data/rli_public_set'
        os.makedirs(local_dir, exist_ok=True)
        
        path = snapshot_download(
            repo_id=repo_id, 
            repo_type='dataset',
            local_dir=local_dir
        )
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def process_rli_data(base_path):
    """Extract instruction entropy and economic metadata from RLI tasks."""
    tasks = []
    # Identify directories public_001 to public_010
    for foldername in os.listdir(base_path):
        if foldername.startswith('public_'):
            folder_path = os.path.join(base_path, foldername)
            
            # Paths to key files
            brief_path = os.path.join(folder_path, 'project', 'brief.md')
            metadata_path = os.path.join(folder_path, 'project', 'metadata.json')
            deliverable_dir = os.path.join(folder_path, 'human_deliverable')
            
            # 1. Instruction Entropy (E)
            # We need the gold standard output to compare against the brief
            # Since deliverables can be multiple files, we'll concatenate text deliverables
            brief_text = ""
            if os.path.exists(brief_path):
                with open(brief_path, 'r', encoding='utf-8') as f:
                    brief_text = f.read()
            
            # Concatenate deliverable content (simplification for E calculation)
            deliverable_text = ""
            artifact_count = 0
            if os.path.exists(deliverable_dir):
                for root, dirs, files in os.walk(deliverable_dir):
                    for file in files:
                        artifact_count += 1
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                deliverable_text += f.read() + "\n"
                        except Exception:
                            pass # Skip binary or unreadable files

            # Calculate E = TokenCount(S) / TokenCount(B)
            brief_tokens = len(brief_text.split())
            deliverable_tokens = len(deliverable_text.split())
            
            entropy = deliverable_tokens / brief_tokens if brief_tokens > 0 else 0
            
            # 2. Economic Value & Metadata
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            tasks.append({
                'task_id': foldername,
                'instruction_entropy': entropy,
                'artifact_coupling': artifact_count, # Counter for kappa
                'human_completion_time': metadata.get('human_completion_hours', 0),
                'project_cost': metadata.get('total_project_cost', 0),
                'category': metadata.get('category', 'Unknown'),
                'success_label': metadata.get('ai_success', 0) # Ground truth for Logit
            })
    
    df = pd.DataFrame(tasks)
    output_path = 'data/rli_processed.csv'
    df.to_csv(output_path, index=False)
    print(f"Processed RLI data saved to: {output_path}")
    return df

if __name__ == "__main__":
    snapshot_path = download_rli_data()
    if snapshot_path:
        process_rli_data(snapshot_path)
