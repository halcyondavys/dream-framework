# Datasets

The datasets used in DREAM's experimental evaluation are not distributed with this repository. They are publicly available from the UCI Machine Learning Repository and OpenML.

---

## Benchmark Datasets (30 real-world datasets)

Place each dataset file in this folder (`data/`) as a `.csv` file with no header, using comma as separator. The target variable must be the **last column**.

| # | Dataset | Source |
|---|---|---|
| 1 | abalone | [UCI](https://archive.ics.uci.edu/dataset/1/abalone) |
| 2 | airfoil_self_noise | [UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self-noise) |
| 3 | albrecht | [OpenML](https://www.openml.org/d/210) |
| 4 | bank32nh | [OpenML](https://www.openml.org/d/573) |
| 5 | bank8FM | [OpenML](https://www.openml.org/d/572) |
| 6 | wiscoinBreastCancer | [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| 7 | ccpp | [UCI](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant) |
| 8 | china | [OpenML](https://www.openml.org/d/189) |
| 9 | cocomonasa60 | [OpenML](https://www.openml.org/d/1049) |
| 10 | cocomo81 | [OpenML](https://www.openml.org/d/1050) |
| 11 | concrete | [UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| 12 | cpu_act | [OpenML](https://www.openml.org/d/197) |
| 13 | cpu_small | [OpenML](https://www.openml.org/d/227) |
| 14 | delta_ailerons | [OpenML](https://www.openml.org/d/179) |
| 15 | delta_elevators | [OpenML](https://www.openml.org/d/198) |
| 16 | desharnais | [OpenML](https://www.openml.org/d/184) |
| 17 | energy_efficiency | [UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| 18 | friedman | [OpenML](https://www.openml.org/d/564) |
| 19 | housing | [OpenML](https://www.openml.org/d/531) |
| 20 | kin8nm | [OpenML](https://www.openml.org/d/189) |
| 21 | machine | [UCI](https://archive.ics.uci.edu/dataset/29/computer+hardware) |
| 22 | maxwell | [OpenML](https://www.openml.org/d/195) |
| 23 | nasa93 | [OpenML](https://www.openml.org/d/1046) |
| 24 | parkinsons_updrs | [UCI](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) |
| 25 | puma32H | [OpenML](https://www.openml.org/d/308) |
| 26 | puma8NH | [OpenML](https://www.openml.org/d/307) |
| 27 | stocks | [OpenML](https://www.openml.org/d/223) |
| 28 | triazines | [UCI](https://archive.ics.uci.edu/dataset/69/triazines) |
| 29 | wineq-red | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| 30 | wineq-white | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) |

---

## Synthetic Datasets (factorial ablation study)

The 12 synthetic datasets used in the factorial ablation study (Appendix B of the thesis) are generated programmatically. No external download is required.

To generate them, run:

```bash
python data/Gerador_synthetic.py
```

The generated files will be saved in `data/synthetic/`. The datasets follow a 2×2×3 factorial design with three structural factors:

- **F1:** Regime discretization (gating sharpness κ) — continuous vs. discrete
- **F2:** Competence dispersion between models — low vs. high
- **F3:** Competence estimation noise — none (F3-baseline), irrelevant attributes (F3a), reduced sample size (F3b)

| Cell | F1 | F2 | F3 | Filename |
|---|---|---|---|---|
| C01 | Continuous | Low | Baseline | C01-continuo-baixo-baixo |
| C02 | Continuous | Low | F3a | C02-continuo-baixo-f3a |
| C03 | Continuous | Low | F3b | C03-continuo-baixo-f3b |
| C04 | Continuous | High | Baseline | C04-continuo-alto-baixo |
| C05 | Continuous | High | F3a | C05-continuo-alto-f3a |
| C06 | Continuous | High | F3b | C06-continuo-alto-f3b |
| C07 | Discrete | Low | Baseline | C07-discreto-baixo-baixo |
| C08 | Discrete | Low | F3a | C08-discreto-baixo-f3a |
| C09 | Discrete | Low | F3b | C09-discreto-baixo-f3b |
| C10 | Discrete | High | Baseline | C10-discreto-alto-baixo |
| C11 | Discrete | High | F3a | C11-discreto-alto-f3a |
| C12 | Discrete | High | F3b | C12-discreto-alto-f3b |
