# -*- coding: utf-8 -*-
"""
Script 09 — Boxplots | Comparação D: DREAM vs. MINE (SizeL=10)
===============================================================
Métodos DREAM : DREAM-DS, DREAM-DW, DREAM-DWS
Métodos MINE  : MINE-DS, MINE-DW, MINE-DWS  (ensemble homogêneo, SizeL=10)
Datasets      : 30
Métrica       : MSE (única disponível no arquivo MINE)

Diferença metodológica
----------------------
DREAM usa ensembles heterogêneos (9 regressores distintos) com seleção global
multicritério na Fase 1. MINE usa ensembles homogêneos com tamanho fixo SizeL.
A comparação avalia se a heterogeneidade e a seleção dinâmica do DREAM
conferem vantagem preditiva sobre a seleção dinâmica homogênea do MINE.

Uso
---
python 09_boxplots_compD_mine10.py
python 09_boxplots_compD_mine10.py --dpi 150 --output-dir outputs/compD10_boxplots
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib as mpl

# Caminhos estruturais do projeto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE3_RESULTS_DIR = RESULTS_DIR / "PHASE3"

# Localiza shared_base.py no mesmo diretório do script
sys.path.insert(0, str(SCRIPT_DIR))

from shared_base import (
    FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN,
    method_label, method_color, sanitize_filename,
    save_descriptive_summary, mpl,
)
from matplotlib.lines import Line2D

LABEL       = "dream_x_mine10"
DESCRIPTION = "Comparação D — DREAM vs. MINE SizeL=10 (30 datasets, métrica: MSE)"
SIZEL       = 10

DREAM_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS"]
MINE_TECNICAS = ["DS", "DW", "DWS"]       # nomes originais no CSV do MINE
MINE_METHODS  = ["MINE_DS", "MINE_DW", "MINE_DWS"]   # nomes após renomeação
ALL_METHODS   = DREAM_METHODS + MINE_METHODS

# Registrar rótulos e cores para métodos MINE no dicionário base
from shared_base import METHOD_DISPLAY_LABELS, METHOD_COLORS
METHOD_DISPLAY_LABELS.update({
    "MINE_DS":  "MINE-DS",
    "MINE_DW":  "MINE-DW",
    "MINE_DWS": "MINE-DWS",
})
METHOD_COLORS.update({
    "MINE_DS":  "#5D6D7E",
    "MINE_DW":  "#839192",
    "MINE_DWS": "#ABB2B9",
})

INPUT_DREAM = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_MINE  = ( PHASE3_RESULTS_DIR / "MINE_Phase3_4_ConsolidatedResults.csv")

def load_dream(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Method"].isin(DREAM_METHODS)][["Dataset","Exec","Fold","Method","MSE"]].copy()
    df = df.rename(columns={"MSE": "Erro"})
    return df


def load_mine(path: Path, sizel: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df[(df["SizeL"] == sizel) & (df["Tecnica"].isin(MINE_TECNICAS))].copy()
    df["Method"] = "MINE_" + df["Tecnica"]
    df = df.rename(columns={"Execucao": "Exec"})[["Dataset","Exec","Fold","Method","Erro"]]
    return df


def plot_boxplot_mse(
    df: pd.DataFrame,
    methods: list[str],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    """Boxplot para MSE — única métrica disponível na comparação DREAM x MINE."""
    agg = (df.groupby(["Dataset","Method"], as_index=False)["Erro"].mean())
    values         = [agg.loc[agg["Method"]==m,"Erro"].dropna().to_numpy() for m in methods]
    display_labels = [method_label(m) for m in methods]
    colors         = [method_color(m, i) for i, m in enumerate(methods)]

    import numpy as np
    fig_w = max(FIGURE_WIDTH_IN, 0.85 * len(methods) + 1.5)
    fig, ax = mpl.pyplot.subplots(figsize=(fig_w, FIGURE_HEIGHT_IN))

    bp = ax.boxplot(
        values, tick_labels=display_labels,
        showmeans=True, meanline=True, patch_artist=True, widths=0.5,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="#E07B00", linewidth=1.5),
        meanprops=dict(color="#2A7A2A", linewidth=1.2, linestyle="--"),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black",
                        markeredgewidth=0.7, markersize=4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Linha divisória visual entre grupos DREAM e MINE
    ax.axvline(x=len(DREAM_METHODS) + 0.5, color="gray",
               linewidth=0.8, linestyle=":", zorder=1)
    ax.text(len(DREAM_METHODS)/2 + 0.5, ax.get_ylim()[1] * 0.97,
            "DREAM", ha="center", fontsize=8.5, color="#2A3A4A",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    ax.text(len(DREAM_METHODS) + len(MINE_METHODS)/2 + 0.5, ax.get_ylim()[1] * 0.97,
            f"MINE (SizeL={SIZEL})", ha="center", fontsize=8.5, color="#2A3A4A",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

    legend_elements = [
        Line2D([0],[0], color="#E07B00", linewidth=1.5, label="Mediana"),
        Line2D([0],[0], color="#2A7A2A", linewidth=1.2, linestyle="--", label="Média"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9)

    ax.set_xlabel("Método"); ax.set_ylabel("MSE")
    ax.tick_params(axis="x", rotation=30)

    if log_scale:
        import numpy as np
        pos = np.concatenate([v[v>0] for v in values if len(v[v>0])>0])
        if len(pos) > 0:
            ax.set_yscale("log"); ax.set_ylabel("MSE (escala log)")

    fig.tight_layout()
    fig.savefig(output_path)
    mpl.pyplot.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--dream",       type=Path, default=INPUT_DREAM)
    p.add_argument("--mine",        type=Path, default=INPUT_MINE)
    p.add_argument("--output-dir",  type=Path, default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_boxplots")
    p.add_argument("--dpi",         type=int,  default=600)
    p.add_argument("--log-scale",   action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams["savefig.dpi"] = args.dpi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LABEL}] {DESCRIPTION}")
    df_dream = load_dream(args.dream)
    df_mine  = load_mine(args.mine, sizel=SIZEL)
    df = pd.concat([df_dream, df_mine], ignore_index=True)

    print(f"  Datasets: {df['Dataset'].nunique()} | Métodos: {', '.join(ALL_METHODS)}")

    out = args.output_dir / "boxplot_MSE.png"
    plot_boxplot_mse(df, methods=ALL_METHODS, output_path=out,
                     log_scale=args.log_scale)
    print(f"  [OK] {out}")

    # Resumo descritivo com coluna renomeada para compatibilidade
    df_desc = df.rename(columns={"Erro": "MSE"})
    save_descriptive_summary(df_desc, metrics=["MSE"], methods=ALL_METHODS,
                             output_dir=args.output_dir)
    print("  [OK] descriptive_summary.csv")


if __name__ == "__main__":
    main()
