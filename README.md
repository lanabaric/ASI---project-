# ASI---project-
ASI project: replication of Dropout as a Bayesian Approximation

This repository contains the code and report for reproducing key experiments from:

**Gal & Ghahramani (2016)**  
*Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*  
[ICML 2016 Paper](https://arxiv.org/abs/1506.02142)

---
This project explores the use of **Monte Carlo Dropout** (MC Dropout) to estimate model uncertainty across four domains:

- **Regression**: Predicting CO₂ levels with uncertainty bands
- **Classification**: Rotated MNIST digit with predictive uncertainty
- **Reinforcement Learning**: Uncertainty-aware exploration (Thompson vs. ε-greedy)
- **Predictive Performance Benchmarks**: RMSE and log-likelihood evaluation

---

## Contents

- `src/` – Python scripts for each experiment
- `notebooks/` – Combined Jupyter notebook version
- `report/` – Final NeurIPS-style report (`.tex` and `.pdf`)
- `results/` – Output plots and figures
- `requirements.txt` – Python dependencies

---

## Getting Started

```bash
git clone https://github.com/lanabaric/ASI---project-.git
cd ASI---project-
conda create -n mc_dropout python=3.10
conda activate mc_dropout
pip install -r requirements.txt
```

---
## Reproducing results

Each experiment is modular:

Run train_regression.py for CO₂ predictions

Run classify_uncertainty.py to evaluate MC Dropout on rotated MNIST

Run rl_uncertainty.py to compare exploration strategies

Run evaluate_mc_dropout.py for benchmark metrics (use smaller sample sizes if slow)
