# Instruction Entropy: A Methodological Proposal for Mapping the High-Entropy Regime (2026)

**Quantifying the structural limits of AI productivity via benchmark selection bias.**

This repository contains the exploratory data pipeline and econometric models used to map the **"High-Entropy Regime"**---the informational boundary where task complexity causes a collapse in AI Marginal Productivity.

## Core Concepts

*   **Inference Density (E):** An Information-Theoretic measure of "Hidden Requirements." Defined as the expansion ratio between the Minimum Description Length (MDL) of the instruction set and the resulting solution.
*   **Coordination Complexity (kappa):** A normalized reference-density metric quantifying state-dependency across multiple solution assets.
*   **Benchmark Curation Bias:** The primary discovery of this pilot study ($p=0.03$). Data suggests that existing AI benchmarks are systematically biased toward modular tasks, masking the true coordination penalties of real-world professional labor.

## Why This Matters

This selection bias implies that **current AI capabilities are systematically overestimated** in domains requiring high coordination complexity. Benchmark performance does not generalize to multi-asset professional tasks, explaining the persistent wage premiums for human experts despite reported AI "parity" on standardized measures.

## Research Methodology (Pilot Study)

*   **Data:** Scale AI RLI (Remote Labor Index) Public Set + O*NET Baseline Wages.
*   **Selection Correction:** A two-stage Heckman procedure utilizing a Probit model to account for non-random task inclusion in professional benchmarks.
*   **Econometrics:** Mean-Centered Translog Production Function with **Wild Cluster Bootstrap** estimation to generate robust inference from finite samples ($G=10$ projects, $N=57$ valid subtasks).
*   **Findings:** Provides significant evidence ($p=0.03$) of benchmark selection bias. While exploratory, the results identify a trend toward non-linear coordination penalties in high-entropy regimes.

## Repository Content

The `output/` directory contains high-fidelity econometric assets:
*   **The Technological Frontier (KDE):** A continuous gradient map of labor distribution.
*   **The Selection Cliff:** Visualization of benchmark curation bias ($p=0.03$).
*   **Translog Results:** Full model coefficients with bootstrapped confidence intervals.

---
*Exploratory Pilot Study by Michael Hernandez*
*Founder, Plethora Solutions, LLC*
