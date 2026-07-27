# DREAM Framework

**DREAM: A Multi-Criteria Global Selection and Local Competence Framework for Dynamic Combination in Heterogeneous Regression Ensembles**

DREAM is a three-phase framework for dynamic ensemble selection and combination applied to regression tasks using heterogeneous pools of base models. It was proposed and evaluated as part of a doctoral thesis in Computer Engineering at PPGEc-UPE.

---

## Framework Overview

DREAM operates in three sequential phases:

**Phase 1 — Global Model Selection**

The first phase builds a heterogeneous pool of base regressors and selects the most promising models using a multicriteria scoring function called CSS (Combined Selection Score). The CSS integrates accuracy, stability, diversity, consensus and a regression-adapted double-fault measure, with weights optimized per dataset via Bayesian search (Optuna/TPE). A dynamic threshold over the score distribution determines the pool size adaptively, retaining between 3 and 9 models. The output is a compact, structurally diverse subset of models passed to the next phases.

**Phase 2 — Local Competence Estimation**

For each test instance, DREAM estimates the local competence of each selected model based on its prediction error in the neighborhood of that instance. Neighborhood is defined using KNN with Inverse Distance Weighting (IDW), so closer training instances contribute more to the competence estimate. Two K-selection strategies are supported: a fixed value (K = 10) or an adaptive heuristic based on training set size (√N for N ≤ 1000; ln(N) for N > 1000, bounded between 3 and 20).

**Phase 3 — Dynamic Combination**

Using the competence estimates from Phase 2, DREAM applies one of three dynamic combination strategies to produce the final prediction:

- **DREAM-DS**: selects models above a competence threshold (λ = 0.3) and combines them using competence-proportional weights, with global MSE penalization for consistently poor models.
- **DREAM-DW**: combines all selected models using L1-normalized competence weights, without threshold filtering.
- **DREAM-DWS**: selects models above a threshold (λ = 0.5) and combines them using uniform weights (1/n selected).

---

## Requirements

- Python >= 3.9

Main dependencies:

```
numpy
pandas
scikit-learn
scipy
optuna
xgboost
matplotlib
seaborn
scikit-posthocs
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

DREAM was evaluated on 30 benchmark regression datasets from the UCI Machine Learning Repository, Torgo, Delve and PROMISE. The datasets are distributed with this repository. and can be downloaded files in `data/real/`. See [`data/README.md`](data/README.md) for the complete list, sources, and download instructions.

All datasets follow the same format: CSV with no header, comma-separated, with the **target variable in the first column**.

The 12 synthetic datasets used in the factorial ablation study are already available in `data/synthetic/` and require no external download. They can also be regenerated at any time by running `data/Gerador_synthetic.py`.

---

## How to Run

### Option 1 — Automatic (recommended)

Runs the complete pipeline in a single command:

```bash
python run_all.py
```

### Option 2 — Manual (step by step)

**Phase 0 — Preprocessing**
```bash
python phase0_dataset/preAnaliseDataset.py
```

**Phase 1 — Global Model Selection**
```bash
python phase1_global/runPhase1_-_Dream.py
python phase1_global/runPhase1_-_Results.py
python phase1_global/runPhase1_Analysis.py
```

**Phase 2 — Local Competence Estimation**
```bash
python phase2_competence/runPhase2.py
python phase2_competence/runPhase2_results.py
```

**Phase 3 — Dynamic Combination**
```bash
python phase3_dynamic/runPhase3.py
python phase3_dynamic/runPhase3_Baseline.py
```

**Phase 3 — Statistical Analysis**

Execute in order (01 to 12):
```bash
python phase3_statical/01_boxplots_phase3.py
python phase3_statical/02_statistical_tests_phase3.py
python phase3_statical/03_boxplots_compA.py
python phase3_statical/04_stats_compA.py
python phase3_statical/05_boxplots_compB.py
python phase3_statical/06_stats_compB.py
python phase3_statical/07_boxplots_compC_.py
python phase3_statical/08_stats_compC_.py
python phase3_statical/09_boxplots_compD_mine10.py
python phase3_statical/10_stats_compD_mine10.py
python phase3_statical/11_boxplots_compD_mine100.py
python phase3_statical/12_stats_compD_mine100.py
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this framework in your research, please consider citing the following works:

**Doctoral Thesis**

Halcyon Davys P. de Carvalho. *DREAM: A Multi-Criteria Global Selection and Local Competence Framework for Dynamic Combination in Heterogeneous Regression Ensembles*. Doctoral thesis, PPGEc-UPE / CIn-UFPE, Recife, Brazil, 2026.

```bibtex
@phdthesis{carvalho2026dream,
  author  = {Halcyon Davys Pimentel de Carvalho},
  title   = {DREAM: A Multi-Criteria Global Selection and Local Competence
             Framework for Dynamic Combination in Heterogeneous Regression Ensembles},
  school  = {Universidade de Pernambuco / Universidade Federal de Pernambuco},
  year    = {2026},
  address = {Recife, Brazil},
  url     = {https://github.com/halcyondavys/dream-framework}
}
```

**Related Article — Systematic Literature Review**

Halcyon Davys P. de Carvalho, João Fausto L. de Oliveira, Roberta Andrade de A. Fagundes. *Dynamic Selection of Ensemble-Based Regression Models: A Systematic Literature Review*. Expert Systems with Applications, 2025.

```bibtex
@article{carvalho2025slr,
  author  = {de Carvalho, Halcyon Davys P. and
             de Oliveira, João Fausto L. and
             de A. Fagundes, Roberta Andrade},
  title   = {Dynamic Selection of Ensemble-Based Regression Models:
             A Systematic Literature Review},
  journal = {Expert Systems with Applications},
  volume  = {290},
  pages   = {128429},
  year    = {2025},
  doi     = {10.1016/j.eswa.2025.128429}
}
```

**Related Article — CBIC 2025**
 
Halcyon Davys P. de Carvalho, João Fausto L. de Oliveira, Roberta Andrade de A. Fagundes. *Ensembles for Regression: A Statistical Analysis and Recommendations*. In: XVII Congresso Brasileiro de Inteligência Computacional (CBIC 2025), Belo Horizonte, Brazil, 2025.
 
```bibtex
@inproceedings{carvalho2025cbic,
  author    = {de Carvalho, Halcyon Davys P. and
               de Oliveira, João Fausto L. and
               de A. Fagundes, Roberta Andrade},
  title     = {Ensembles for Regression: A Statistical Analysis and Recommendations},
  booktitle = {Anais do XVII Congresso Brasileiro de Intelig{\^e}ncia Computacional (CBIC 2025)},
  year      = {2025},
  address   = {Belo Horizonte, Brazil},
  publisher = {SBIC},
  doi       = {10.21528/CBIC2025-1176303}
}
```

---

**Halcyon Davys Pimentel de Carvalho**
ORCID: [0000-0001-8933-5912](https://orcid.org/0000-0001-8933-5912)
PPGEc-UPE / CIn-UFPE — Recife, Brazil
