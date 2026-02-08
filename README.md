# Instruction Entropy: Quantifying the High-Entropy Regime (2026)

**A methodological framework for identifying the structural limits of AI productivity in multi-asset professional environments.**

This repository contains the exploratory data pipeline and econometric models used to map the "High-Entropy Regime"---the informational coordinate where task complexity causes a non-linear collapse in AI Marginal Productivity.

## Core Concepts

*   **Inference Density (E):** An Information-Theoretic measure of "Hidden Requirements." Defined as the expansion ratio between the Minimum Description Length (MDL) of the instruction set and the resulting solution.
*   **Coordination Complexity (kappa):** A normalized reference-density metric quantifying state-dependency across multiple solution assets.
*   **The Structural Break:** The non-linear threshold where the cost of AI orchestration begins to exceed the value of execution, defining the 2026 human labor floor.

## Research Methodology (Pilot Study)

*   **Data:** Scale AI RLI (Remote Labor Index) Public Set + O*NET Baseline Wages.
*   **Selection Correction:** A two-stage Heckman procedure utilizing 'Automation Exposure' as an instrument to account for non-random task inclusion in benchmarks.
*   **Econometrics:** Clustered Hedonic Translog Production Function with **Wild Cluster Bootstrap-t** estimation to generate robust inference from finite samples (G=10 projects, N=156 requirements).

## Repository Content

The `output/` directory contains high-fidelity econometric assets:
*   **The Technological Frontier (KDE):** A continuous gradient map of labor distribution across inference and coordination axes.
*   **Selection Response Surface:** Mapping the probability of task inclusion vs. automation exposure.
*   **Structural Break Analysis:** Visualizing the non-linear shift in wage elasticity.

## License and Trademarks

Copyright 2026 Michael Hernandez. All rights reserved.

The source code in this repository is licensed under the **Apache License, Version 2.0**. See the `LICENSE` file for details.

**Trademark Notice:** The terms **"Instruction Entropy"** and **"The Complexity Kink"** are trademarks of Michael Hernandez / Plethora Solutions, LLC.

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

### Running the Analysis
1. **Retrieve Data:** `python src/ingestion_rli.py`
2. **Build Master Dataset:** `python src/process_master.py`
3. **Execute Scientific Model:** `python src/final_scientific_model.py`

## Robustness and Defense

1. **Construct Validity of Instruction Entropy (E):**
   - *Critique:* E is sensitive to prompt verbosity.
   - *Defense:* I apply MDL log-smoothing to instructions and a boilerplate-mask to solution assets, ensuring E measures "Inference Density" rather than "Wordiness."

2. **The Agentic Loop Counter-Argument:**
   - *Critique:* Agentic workflows solve Artifact Coupling.
   - *Defense:* The Translog quadratic term proves a non-linear coordination cliff. Breaking one complex task into 1,000 sub-tasks merely shifts complexity from execution to orchestration.

3. **Small Sample Size (N=10):**
   - *Critique:* Standard errors are too high for significance.
   - *Defense:* I decompose 10 projects into 156 requirements and utilize Clustered Standard Errors to maintain scientific rigor while increasing statistical power.

4. **Self-Reporting Bias in Wage Data:**
   - *Critique:* Freelance time-tracking is noisy.
   - *Defense:* I transition to a Market Equilibrium Value anchored to O*NET SOC-specific baseline wages and apply 5% Winsorization to remove extreme outliers.

## Limitations and Future Work

*   **Selection Bias:** The public RLI tasks were curated by Scale AI. Future research should apply this E-kappa framework to wider, non-benchmarked freelance datasets to verify the universal stability of the Kink coordinates.
*   **Dynamic Kink:** The coordinates of the Complexity Kink are likely moving rightward as model reasoning improves. This study represents a snapshot of the labor floor in early 2026.
*   **Instruction Clarity Paradox:** While log-smoothing mitigates the effect of vague briefs, Instruction Entropy remains partially dependent on the skill of the human "Prompt Engineer." Further research is needed to isolate "Task Complexity" from "Instruction Quality."

## Citation
Mazeika, M., et al. (2025). Remote Labor Index: Measuring AI Automation of Remote Work. arXiv:2510.26787.

---
*Research by Michael Hernandez*
