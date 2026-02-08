# Instruction Entropy: Quantifying the High-Entropy Regime (2026)

**A methodological framework for identifying the structural limits of AI productivity in multi-asset professional environments.**

This repository contains the exploratory data pipeline and econometric models used to map the "High-Entropy Regime"---the informational coordinate where task complexity causes a non-linear collapse in AI Marginal Productivity.

## Core Concepts

*   **Inference Density (E):** An Information-Theoretic measure of "Hidden Requirements." Defined as the expansion ratio between the Minimum Description Length (MDL) of the instruction set and the resulting solution.
*   **Coordination Complexity (kappa):** A normalized reference-density metric quantifying state-dependency across multiple solution assets.
*   **The Structural Break:** The non-linear threshold where the cost of AI orchestration begins to exceed the value of execution, defining the 2026 human labor floor.

## Research Methodology (Pilot Study)

*   **Data:** Scale AI RLI (Remote Labor Index) Public Set + O*NET Baseline Wages.
*   **Selection Correction:** A two-stage Heckman procedure utilizing 'Automation Exposure' as an instrument to account for non-random task inclusion in professional benchmarks.
*   **Econometrics:** Clustered Hedonic Translog Production Function with **Wild Cluster Bootstrap-t** estimation to generate robust inference from finite samples (G=10 projects, N=156 requirements).

## Repository Content

The `output/` directory contains high-fidelity econometric assets:
*   **The Technological Frontier (KDE):** A continuous gradient map of labor distribution across inference and coordination axes.
*   **Selection Response Surface:** Mapping the probability of task inclusion vs. automation exposure.
*   **Structural Break Analysis:** Visualizing the non-linear shift in wage elasticity.

## Setup and Execution

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Installation
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Hardened Analysis (v2)
1. **Retrieve Data:** `python src/ingestion_rli.py`
2. **Decompose Requirements:** `python src/decomposition_layer_v2.py`
3. **Execute Scientific Model:** `python src/final_scientific_model_v2.py`

## Robustness and Defense

1. **Construct Validity of Inference Density (E):**
   - *Critique:* E is sensitive to prompt verbosity.
   - *Defense:* I transition from token-counts to **MDL Expansion Ratios** (via zlib compression). This ensures E measures pure informational expansion rather than "wordiness."

2. **The Small-N Cluster Problem:**
   - *Critique:* Standard errors are unreliable with only 10 clusters.
   - *Defense:* I implement a **Wild Cluster Bootstrap** procedure to generate robust critical values, minimizing Type I error inflation inherent in small project counts.

3. **Sample Selection Bias:**
   - *Critique:* Professional benchmarks preferentially include "feasible" tasks.
   - *Defense:* I utilize a **Heckman Two-Stage Correction**. The first-stage Probit (significant at p < 0.001) confirms that 'Automation Exposure' is a valid instrument for modeling the probability of task inclusion.

## Limitations and Future Work

*   **Exploratory Sample:** This study represents a pilot methodology. Future research must expand the E-kappa framework to wider, non-benchmarked datasets to verify the universal stability of the structural break.
*   **The Instruction Quality Paradox:** While MDL ratios mitigate prompt variance, the labor required to generate high-signal instructions represents a shift from 'Execution' to 'Orchestration.' The "High-Entropy Regime" identifies the coordinate where this tradeoff becomes economically non-viable.

## Citation
Mazeika, M., et al. (2025). Remote Labor Index: Measuring AI Automation of Remote Work. arXiv:2510.26787.

---
*Research by Michael Hernandez*
