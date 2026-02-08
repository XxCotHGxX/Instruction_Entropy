# Instruction Entropy: A Methodological Proposal for Mapping the High-Entropy Regime (2026)

**An exploratory framework for identifying the structural limits of AI productivity in multi-asset professional environments.**

This repository contains the exploratory data pipeline and econometric models used to map the "High-Entropy Regime"---the informational boundary where task complexity causes a non-linear collapse in AI Marginal Productivity.

## Core Concepts

*   **Inference Density (E):** An Information-Theoretic measure of "Hidden Requirements." Defined as the expansion ratio between the Minimum Description Length (MDL) of the instruction set and the resulting solution.
*   **Coordination Complexity (kappa):** A normalized reference-density metric quantifying state-dependency across multiple solution assets.
*   **The Structural Break:** The informational coordinate where the cost of AI orchestration begins to exceed the value of execution, defining the 2026 human labor floor.

## Research Methodology (Pilot Study)

*   **Data:** Scale AI RLI (Remote Labor Index) Public Set + O*NET Baseline Wages.
*   **Selection Correction:** A two-stage Heckman procedure utilizing a Probit model to account for non-random task inclusion in professional benchmarks.
*   **Econometrics:** Mean-Centered Translog Production Function with **Wild Cluster Bootstrap** estimation to generate robust inference from finite samples (G=10 projects, N=57 valid subtasks).
*   **Validation:** Out-of-sample predictive testing performed against the **CascadingLight** market lead dataset.

## Repository Content

The `output/` directory contains high-fidelity econometric assets:
*   **The Technological Frontier (KDE):** A continuous gradient map of labor distribution across inference and coordination axes.
*   **Selection Stage Results:** Probit coefficients for task inclusion feasibility.
*   **Translog Results:** Full model coefficients with bootstrapped confidence intervals.

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
2. **Hardened Decomposition:** `python src/decomposition_layer_v2.py`
3. **Hardened Model (v3):** `python src/final_scientific_model_v2.py`

## Robustness and Defense

1. **Construct Validity of Inference Density (E):**
   - *Defense:* I transition from unit-dependent proxies to unit-less **MDL Expansion Ratios**. This ensures E measures pure informational expansion.

2. **Identification Strategy:**
   - *Defense:* I utilize a **Heckman Two-Stage Correction** to address selection bias in benchmarks. The first-stage Probit (z=7.34, p < 0.001) confirms the validity of the selection instrument.

3. **Small-N Cluster Rigor:**
   - *Defense:* I implement a **Wild Cluster Bootstrap** to generate robust confidence intervals, minimizing Type I error risks associated with small cluster counts (G=10).

---
*Exploratory Pilot Study by Michael Hernandez*
*Founder, Plethora Solutions, LLC*
