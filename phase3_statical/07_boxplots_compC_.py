# -*- coding: utf-8 -*-
"""
Script 07 — Boxplots | Comparação B
=====================================
Métodos : DREAM-DS, DREAM-DW, DREAM-DWS, TOP3-AVG, TOP5-AVG, HET-DREAM
Datasets: 9 (apenas datasets com N_Models >= 5 selecionados na Fase 1)
          bank8FM, china, cocomo81, delta_ailerons, delta_elevators,
          desharnais, machine, maxwell, wiscoinBreastCancer

Justificativa do filtro
-----------------------
TOP5-AVG só pode ser calculado quando a Fase 1 selecionou ao menos 5 modelos.
O filtro garante comparação homogênea entre todos os métodos.

Uso
---
python 07_boxplots_compC_.py
python 07_boxplots_compC_.py --dpi 150
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Caminhos estruturais do projeto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE3_RESULTS_DIR = RESULTS_DIR / "PHASE3"

# Localiza shared_base.py no mesmo diretório do script
sys.path.insert(0, str(SCRIPT_DIR))

from shared_base import (
    DEFAULT_METRICS, plot_global_boxplot, load_and_merge,
    save_descriptive_summary, mpl,
)

LABEL          = "dream_x_top5"
DESCRIPTION    = "Comparação TOP5 — DREAM vs. TOP3/TOP5/HET-DREAM (9 datasets, N_Models≥5)"
ACTIVE_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS", "TOP3_AVG", "TOP5_AVG", "HET_DREAM_AVG"]

# Datasets com N_Models >= 5 identificados pela Fase 1 (TOP_ALL_AVG como referência)
DATASETS_B = [
    "bank8FM", "china", "cocomo81", "delta_ailerons", "delta_elevators",
    "desharnais", "machine", "maxwell", "wiscoinBreastCancer",
]

INPUT_PHASE3    = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_BASELINES = ( PHASE3_RESULTS_DIR / "dream_phase3_baselines_results.csv" )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--phase3",    type=Path, default=INPUT_PHASE3)
    p.add_argument("--baselines", type=Path, default=INPUT_BASELINES)
    p.add_argument("--output-dir", type=Path, default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_boxplots")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--log-scale", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams["savefig.dpi"] = args.dpi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LABEL}] {DESCRIPTION}")
    df = load_and_merge([args.phase3, args.baselines],
                        methods=ACTIVE_METHODS, datasets=DATASETS_B)
    n_ds = df["Dataset"].nunique()
    print(f"  Datasets carregados: {n_ds} (esperado: {len(DATASETS_B)})")
    print(f"  Métodos: {', '.join(ACTIVE_METHODS)}")

    if n_ds < len(DATASETS_B):
        missing = set(DATASETS_B) - set(df["Dataset"].unique())
        print(f"  [AVISO] Datasets ausentes: {missing}")

    metrics = [m for m in args.metrics if m in df.columns]

    for metric in metrics:
        out = args.output_dir / f"boxplot_{metric}.png"
        plot_global_boxplot(df, metric=metric, methods=ACTIVE_METHODS,
                            output_path=out, log_scale=args.log_scale)
        print(f"  [OK] {out}")

    save_descriptive_summary(df, metrics=metrics, methods=ACTIVE_METHODS,
                             output_dir=args.output_dir)
    print(f"  [OK] descriptive_summary.csv")


if __name__ == "__main__":
    main()
