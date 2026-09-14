# Datasets

The datasets used in DREAM's experimental evaluation are available in this repository. The real-world benchmark datasets are stored in `data/real/`, whereas the synthetic datasets used in the factorial ablation study are stored in `data/synthetic/`.. They are publicly available from the UCI Machine Learning Repository and OpenML.

---

## Benchmark Datasets

The main experimental evaluation comprises 30 benchmark regression datasets. The dataset identifiers and original sources are listed below.

The real-world datasets are stored in `data/real/` as `.data` files with no header, using commas as separators. The target variable must be placed in the **first column**, and the remaining columns correspond to the input features.

| # | Dataset | Repository | Source |
|---:|---|---|---|
| 1 | abalone | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/1/abalone) |
| 2 | airfoil_self_noise | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/291/airfoil+self-noise) |
| 3 | albrecht | OpenML | [Dataset page](https://www.openml.org/d/210) |
| 4 | bank32nh | OpenML | [Dataset page](https://www.openml.org/d/573) |
| 5 | bank8FM | OpenML | [Dataset page](https://www.openml.org/d/572) |
| 6 | wiscoinBreastCancer | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| 7 | ccpp | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) |
| 8 | china | [Zenodo] | [Dataset page](https://doi.org/10.5281/zenodo.268446) |
| 9 | cocomonasa60 | OpenML | [Dataset page](https://www.openml.org/d/1049) |
| 10 | cocomo81 | OpenML | [Dataset page](https://www.openml.org/d/1050) |
| 11 | concrete | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| 12 | cpu_act | OpenML | [Dataset page](https://www.openml.org/d/197) |
| 13 | cpu_small | OpenML | [Dataset page](https://www.openml.org/d/227) |
| 14 | delta_ailerons | OpenML | [Dataset page](https://www.openml.org/d/179) |
| 15 | delta_elevators | OpenML | [Dataset page](https://www.openml.org/d/198) |
| 16 | desharnais | OpenML | [Dataset page](https://www.openml.org/d/184) |
| 17 | energy_efficiency | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| 18 | friedman | OpenML | [Dataset page](https://www.openml.org/d/564) |
| 19 | housing | OpenML | [Dataset page](https://www.openml.org/d/531) |
| 20 | kin8nm | OpenML | [Dataset page](https://www.openml.org/d/189) |
| 21 | machine | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/29/computer+hardware) |
| 22 | maxwell | OpenML | [Dataset page](https://www.openml.org/d/195) |
| 23 | nasa93 | OpenML | [Dataset page](https://www.openml.org/d/1046) |
| 24 | parkinsons_updrs | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) |
| 25 | puma32H | OpenML | [Dataset page](https://www.openml.org/d/308) |
| 26 | puma8NH | OpenML | [Dataset page](https://www.openml.org/d/307) |
| 27 | stocks | OpenML | [Dataset page](https://www.openml.org/d/223) |
| 28 | triazines | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/69/triazines) |
| 29 | wineq-red | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| 30 | wineq-white | UCI | [Dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality) |

---

## Data Format and Loading Convention

All files use a numerical, comma-separated representation without a header row. The first column contains the regression target, and all remaining columns contain the input features.

The datasets are loaded by `phase1_global/loadData.py`. Dataset selection is defined by the `datasets_used()` function in `main/configs.py`.

The preprocessing stage can be executed with:

```bash
python phase0_dataset/preAnaliseDataset.py
```

---

## Synthetic Datasets (factorial ablation study)

The 12 synthetic datasets used in the factorial ablation study (Appendix B of the thesis) are generated programmatically. No external download is required.

To generate them, run:

```bash
python data/Gerador_synthetic.py
```

The generated datasets are available in `data/synthetic/` and can be regenerated using the command above. The datasets follow a 2×2×3 factorial design with three structural factors:

- **F1:** Regime discretization (gating sharpness κ) — continuous vs. discrete
- **F2:** Competence dispersion between models — low vs. high
- **F3:** Competence estimation noise — none (F3-baseline), irrelevant attributes (F3a), reduced sample size (F3b)

| Cell | F1 | F2 | F3 | Filename |
|---|---|---|---|---|
| C01 | Continuous | Low | Baseline | C01-continuo-baixo-baixo.data |
| C02 | Continuous | Low | F3a | C02-continuo-baixo-f3a.data |
| C03 | Continuous | Low | F3b | C03-continuo-baixo-f3b.data |
| C04 | Continuous | High | Baseline | C04-continuo-alto-baixo.data |
| C05 | Continuous | High | F3a | C05-continuo-alto-f3a.data |
| C06 | Continuous | High | F3b | C06-continuo-alto-f3b.data |
| C07 | Discrete | Low | Baseline | C07-discreto-baixo-baixo.data |
| C08 | Discrete | Low | F3a | C08-discreto-baixo-f3a.data |
| C09 | Discrete | Low | F3b | C09-discreto-baixo-f3b.data |
| C10 | Discrete | High | Baseline | C10-discreto-alto-baixo.data |
| C11 | Discrete | High | F3a | C11-discreto-alto-f3a.data |
| C12 | Discrete | High | F3b | C12-discreto-alto-f3b.data |

---

## Dataset Attribution

The datasets retain the attribution and usage conditions established by their original providers. The copies included in this repository are provided to support the reproducibility of the DREAM experiments.

Users intending to redistribute or reuse individual datasets should consult the corresponding UCI, OpenML or Zenodo dataset page for licensing and attribution information.
