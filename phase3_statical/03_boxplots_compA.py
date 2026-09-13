# -*- coding: utf-8 -*-
"""
Script 03 — Boxplots | Comparação A
=====================================
Métodos : DREAM-DS, DREAM-DW, DREAM-DWS, TOP3-AVG, HET-DREAM
Datasets: 30 (todos os datasets reais)
Arquivos: dream_phase3_results.csv + dream_phase3_baselines_results.csv

Uso
---
python 03_boxplots_compA.py
python 03_boxplots_compA.py --dpi 150 --output-dir outputs/compA_boxplots
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
    DEFAULT_METRICS, plot_global_boxplot, load_and_merge, save_descriptive_summary,
    mpl,
)

# ---------------------------------------------------------------------------
# Configuração da comparação
# ---------------------------------------------------------------------------
LABEL         = "dream_x_avg"
DESCRIPTION   = "Comparação A — DREAM vs. TOP3-AVG e HET-DREAM (30 datasets)"
ACTIVE_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS", "TOP3_AVG", "HET_DREAM_AVG"]

INPUT_PHASE3 = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_BASELINES = ( PHASE3_RESULTS_DIR / "dream_phase3_baselines_results.csv" )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--phase3",    type=Path, default=INPUT_PHASE3)
    p.add_argument("--baselines", type=Path, default=INPUT_BASELINES)
    p.add_argument("--output-dir", type=Path, default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_boxplots")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--log-scale", action="store_true", default=False,
                   help="Escala log nos erros. Desabilitado por padrão.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams["savefig.dpi"] = args.dpi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LABEL}] {DESCRIPTION}")
    df = load_and_merge([args.phase3, args.baselines], methods=ACTIVE_METHODS)
    print(f"  Datasets carregados: {df['Dataset'].nunique()}")
    print(f"  Métodos: {', '.join(ACTIVE_METHODS)}")

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
