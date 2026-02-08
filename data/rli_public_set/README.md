---
license: apache-2.0
task_categories:
- other
language:
- en
tags:
- evaluation
- benchmark
size_categories:
- n<1K
viewer: false
---

# rli-public-set

The RLI public set

## Dataset Structure

This dataset contains project folders organized by project ID (public_001 through public_010).

Each project folder contains:
- `human_deliverable/` - Reference deliverables created by human experts
- `project/` - Project specifications and inputs
  - `brief.md` - Project description and requirements
  - `inputs/` - Input files provided for the project

## Usage

```python
from huggingface_hub import snapshot_download

# Download entire dataset
dataset_path = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset"
)

# Access specific project
import os
project_001_path = os.path.join(dataset_path, "public_001")
```

## Citation

If you use this dataset in your research, please include this citation.
```
@misc{mazeika2025remote,
      title = {Remote Labor Index: Measuring AI Automation of Remote Work},
      author = {Mantas Mazeika and Alice Gatti and Cristina Menghini and Udari Madhushani Sehwag and Shivam Singhal and Yury Orlovskiy and Steven Basart and Manasi Sharma and Denis Peskoff and Elaine Lau and Jaehyuk Lim and Lachlan Carroll and Alice Blair and Vinaya Sivakumar and Sumana Basu and Brad Kenstler and Yuntao Ma and Julian Michael and Xiaoke Li and Oliver Ingebretsen and Aditya Mehta and Jean Mottola and John Teichmann and Kevin Yu and Zaina Shaik and Adam Khoja and Richard Ren and Jason Hausenloy and Long Phan and Ye Htet and Ankit Aich and Tahseen Rabbani and Vivswan Shah and Andriy Novykov and Felix Binder and Kirill Chugunov and Luis Ramirez and Matias Geralnik and Hernán Mesura and Dean Lee and Ed-Yeremai Hernandez Cardona and Annette Diamond and Summer Yue and Alexandr Wang and Bing Liu and Ernesto Hernandez and Dan Hendrycks},
      year            = {2025},
      eprint          = {2510.26787},
      archivePrefix   = {arXiv},
      primaryClass    = {cs.LG},
      url             = {https://arxiv.org/abs/2510.26787}
}
```

## License

Apache 2.0
