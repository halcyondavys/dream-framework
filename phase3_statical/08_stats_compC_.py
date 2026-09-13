# -*- coding: utf-8 -*-
"""
Script 08 — Testes estatísticos | Comparação B
================================================
Métodos : DREAM-DS, DREAM-DW, DREAM-DWS, TOP3-AVG, TOP5-AVG, HET-DREAM
Datasets: 9 (N_Models >= 5 na Fase 1)

Observação metodológica
-----------------------
Com N=9 blocos, o poder estatístico do Friedman é reduzido. Resultados
não significativos devem ser discutidos à luz do tamanho amostral limitado.
O CD diagram e o heatmap Nemenyi permanecem válidos como referência visual.

Uso
---
python 08_stats_compC_.py
python 08_stats_compC_.py --wilcoxon --latex
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

from shared_base import DEFAULT_METRICS, load_and_merge, run_statistical_pipeline

LABEL          = "dream_x_top5"
DESCRIPTION    = "Comparação TOP5 — DREAM vs. TOP3/TOP5/HET-DREAM (9 datasets, N_Models≥5)"
ACTIVE_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS", "TOP3_AVG", "TOP5_AVG", "HET_DREAM_AVG"]

DATASETS_B = [
    "bank8FM", "china", "cocomo81", "delta_ailerons", "delta_elevators",
    "desharnais", "machine", "maxwell", "wiscoinBreastCancer",
]

INPUT_PHASE3    = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_BASELINES = ( PHASE3_RESULTS_DIR / "dream_phase3_baselines_results.csv" )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--phase3",          type=Path,  default=INPUT_PHASE3)
    p.add_argument("--baselines",       type=Path,  default=INPUT_BASELINES)
    p.add_argument("--output-dir",      type=Path,  default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_statistic")
    p.add_argument("--metrics",         nargs="+",  default=DEFAULT_METRICS)
    p.add_argument("--alpha",           type=float, default=0.05)
    p.add_argument("--reference-method",type=str,   default="DREAM_DW")
    p.add_argument("--wilcoxon",        action="store_true", default=False)
    p.add_argument("--latex",           action="store_true", default=False)
    p.add_argument("--dpi",             type=int,   default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[{LABEL}] {DESCRIPTION}")
    df = load_and_merge([args.phase3, args.baselines],
                        methods=ACTIVE_METHODS, datasets=DATASETS_B)
    n_ds = df["Dataset"].nunique()
    print(f"  Datasets: {n_ds} | Métodos: {', '.join(ACTIVE_METHODS)}")

    if n_ds < 2:
        raise ValueError("Datasets insuficientes para testes estatísticos.")

    run_statistical_pipeline(
        df=df, methods=ACTIVE_METHODS, metrics=args.metrics,
        output_dir=args.output_dir, alpha=args.alpha,
        wilcoxon_enabled=args.wilcoxon, reference_method=args.reference_method,
        latex_enabled=args.latex, label_prefix=LABEL, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
