# -*- coding: utf-8 -*-
"""
Script 06 — Testes estatísticos | Comparação C
================================================
Métodos : DREAM-DS, DREAM-DW, DREAM-DWS, Stacking-ET, Stacking-RF, Stacking-XGB
Datasets: 30

Enquadramento metodológico
--------------------------
O Stacking opera como teto de complexidade. Mesmo que supere o DREAM
estatisticamente, resultados próximos (diferença de ranking pequena ou
grupos Nemenyi sobrepostos) sustentam a escolha do DREAM com base em:
  - menor custo computacional de inferência;
  - interpretabilidade das estratégias de seleção;
  - ausência de meta-treinamento adicional.

Uso
---
python 06_stats_compB.py
python 06_stats_compB.py --wilcoxon --latex
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

LABEL          = "dream_x_stacking"
DESCRIPTION    = "Comparação B — DREAM vs. Stacking (30 datasets, teto de complexidade)"
ACTIVE_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS",
                  "STACKING_ET", "STACKING_RF", "STACKING_XGB"]

INPUT_PHASE3    = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_BASELINES = ( PHASE3_RESULTS_DIR / "dream_phase3_baselines_results.csv" )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--phase3",           type=Path,  default=INPUT_PHASE3)
    p.add_argument("--baselines",        type=Path,  default=INPUT_BASELINES)
    p.add_argument("--output-dir",       type=Path,  default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_statistic")
    p.add_argument("--metrics",          nargs="+",  default=DEFAULT_METRICS)
    p.add_argument("--alpha",            type=float, default=0.05)
    p.add_argument("--reference-method", type=str,   default="DREAM_DW")
    p.add_argument("--wilcoxon",         action="store_true", default=False)
    p.add_argument("--latex",            action="store_true", default=False)
    p.add_argument("--dpi",              type=int,   default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[{LABEL}] {DESCRIPTION}")
    df = load_and_merge([args.phase3, args.baselines], methods=ACTIVE_METHODS)
    print(f"  Datasets: {df['Dataset'].nunique()} | Métodos: {', '.join(ACTIVE_METHODS)}")

    run_statistical_pipeline(
        df=df, methods=ACTIVE_METHODS, metrics=args.metrics,
        output_dir=args.output_dir, alpha=args.alpha,
        wilcoxon_enabled=args.wilcoxon, reference_method=args.reference_method,
        latex_enabled=args.latex, label_prefix=LABEL, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
