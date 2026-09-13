# -*- coding: utf-8 -*-
"""
Script 10 — Testes estatísticos | Comparação D: DREAM vs. MINE (SizeL=10)
==========================================================================
Métodos   : DREAM-DS, DREAM-DW, DREAM-DWS, MINE-DS, MINE-DW, MINE-DWS
Datasets  : 30
Métrica   : MSE (única disponível no arquivo MINE)
Protocolo : Friedman + Nemenyi (Demšar 2006)

Interpretação esperada
----------------------
O MINE representa o estado da arte em seleção dinâmica com ensembles
homogêneos (Moura 2019). Diferenças de ranking a favor do DREAM sustentam
a hipótese de que ensembles heterogêneos com seleção global multicritério
produzem melhores estimadores locais de competência. Resultados não
significativos indicam paridade, que também é defensável: o DREAM oferece
interpretabilidade sobre a composição do pool sem perda de precisão.

Uso
---
python 10_stats_compD_mine10.py
python 10_stats_compD_mine10.py --wilcoxon --latex
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Caminhos estruturais do projeto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE3_RESULTS_DIR = RESULTS_DIR / "PHASE3"

# Localiza shared_base.py no mesmo diretório do script
sys.path.insert(0, str(SCRIPT_DIR))

from shared_base import (
    METHOD_DISPLAY_LABELS, METHOD_COLORS,
    aggregate_by_dataset, rank_methods, friedman_test,
    paired_wilcoxon_all_pairs, wins_by_dataset, try_nemenyi,
    plot_cd_diagram, plot_nemenyi_heatmap, plot_win_counts,
    save_latex_tables, sanitize_filename, method_label, method_color,
    FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN,
)

# Registrar métodos MINE
METHOD_DISPLAY_LABELS.update({"MINE_DS":"MINE-DS","MINE_DW":"MINE-DW","MINE_DWS":"MINE-DWS"})
METHOD_COLORS.update({"MINE_DS":"#5D6D7E","MINE_DW":"#839192","MINE_DWS":"#ABB2B9"})

LABEL         = "dream_x_mine10"
DESCRIPTION   = "Comparação D — DREAM vs. MINE SizeL=10 (30 datasets, MSE)"
SIZEL         = 10
DREAM_METHODS = ["DREAM_DS", "DREAM_DW", "DREAM_DWS"]
MINE_METHODS  = ["MINE_DS",  "MINE_DW",  "MINE_DWS"]
ALL_METHODS   = DREAM_METHODS + MINE_METHODS
MINE_TECNICAS = ["DS", "DW", "DWS"]

INPUT_DREAM = ( PHASE3_RESULTS_DIR / "dream_phase3_results.csv" )
INPUT_MINE  = ( PHASE3_RESULTS_DIR / "MINE_Phase3_4_ConsolidatedResults.csv")

def load_combined(dream_path: Path, mine_path: Path, sizel: int) -> pd.DataFrame:
    """
    Carrega e combina DREAM (MSE) e MINE (Erro=MSE) em um DataFrame
    padronizado com colunas: Dataset, Exec, Fold, Method, MSE.
    """
    df_dream = pd.read_csv(dream_path)
    df_dream = df_dream[df_dream["Method"].isin(DREAM_METHODS)][
        ["Dataset","Exec","Fold","Method","MSE"]].copy()

    df_mine = pd.read_csv(mine_path, sep=";")
    df_mine = df_mine[(df_mine["SizeL"]==sizel) & (df_mine["Tecnica"].isin(MINE_TECNICAS))].copy()
    df_mine["Method"] = "MINE_" + df_mine["Tecnica"]
    df_mine = df_mine.rename(columns={"Execucao":"Exec","Erro":"MSE"})[
        ["Dataset","Exec","Fold","Method","MSE"]]

    return pd.concat([df_dream, df_mine], ignore_index=True)


def plot_win_counts_mse(
    wins_count_df: pd.DataFrame,
    methods: list[str],
    output_dir: Path,
) -> None:
    """Bar chart de vitórias para MSE com separador visual DREAM | MINE."""
    subset = wins_count_df[wins_count_df["Metric"] == "MSE"].copy()
    if subset.empty:
        return
    subset["_order"] = subset["Best_Method"].apply(
        lambda m: methods.index(m) if m in methods else 99)
    subset = subset.sort_values("_order")

    display_labels = [method_label(m) for m in subset["Best_Method"]]
    colors = [method_color(m, i) for i, m in enumerate(subset["Best_Method"])]
    wins = subset["Wins"].tolist()
    total = sum(wins)

    fig_w = max(FIGURE_WIDTH_IN, 0.85 * len(display_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, FIGURE_HEIGHT_IN))
    bars = ax.bar(display_labels, wins, color=colors, width=0.55,
                  edgecolor="black", linewidth=0.6, alpha=0.85)

    for bar, w in zip(bars, wins):
        pct = w / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.12,
                f"{w}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9)

    ax.axvline(x=len(DREAM_METHODS) - 0.5, color="gray",
               linewidth=0.8, linestyle=":", zorder=0)

    ax.set_xlabel("Método"); ax.set_ylabel("Nº de vitórias")
    ax.set_ylim(0, total + 2)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = output_dir / "win_counts_MSE.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] Win counts: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--dream",            type=Path,  default=INPUT_DREAM)
    p.add_argument("--mine",             type=Path,  default=INPUT_MINE)
    p.add_argument("--output-dir",       type=Path,  default=PHASE3_RESULTS_DIR / "outputs_baselines" / f"{LABEL}_stats")
    p.add_argument("--alpha",            type=float, default=0.05)
    p.add_argument("--reference-method", type=str,   default="DREAM_DW")
    p.add_argument("--wilcoxon",         action="store_true", default=False)
    p.add_argument("--latex",            action="store_true", default=False)
    p.add_argument("--dpi",              type=int,   default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mpl.rcParams["savefig.dpi"] = args.dpi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LABEL}] {DESCRIPTION}")
    df = load_combined(args.dream, args.mine, sizel=SIZEL)
    print(f"  Datasets: {df['Dataset'].nunique()} | Métodos: {', '.join(ALL_METHODS)}")

    metric = "MSE"
    wide = aggregate_by_dataset(df, metric=metric, methods=ALL_METHODS)
    wide.to_csv(args.output_dir / f"matrix_{metric}.csv")

    friedman_row  = friedman_test(wide, metric)
    friedman_df   = pd.DataFrame([friedman_row])
    friedman_df["Significant"] = friedman_df["P_Value"] < args.alpha
    rank_df       = rank_methods(wide, metric)
    wins_df       = wins_by_dataset(wide, metric)
    wins_count_df = (wins_df.groupby(["Metric","Best_Method"], as_index=False)
                     .size().rename(columns={"size":"Wins"})
                     .sort_values("Wins", ascending=False))

    friedman_df.to_csv(args.output_dir / "friedman_summary.csv", index=False)
    rank_df.to_csv(args.output_dir    / "average_ranks.csv",     index=False)
    wins_df.to_csv(args.output_dir    / "best_method_by_dataset.csv", index=False)
    wins_count_df.to_csv(args.output_dir / "win_counts.csv",     index=False)

    print(f"\n  === {metric} ===")
    print(f"  Friedman: χ²={friedman_row['Friedman_Statistic']:.4f}, "
          f"p={friedman_row['P_Value']:.4f}, "
          f"Sig={'Sim' if friedman_df['Significant'].iloc[0] else 'Não'}")

    pairwise_df = None
    if args.wilcoxon:
        pairwise_df = paired_wilcoxon_all_pairs(
            wide=wide, metric=metric, alpha=args.alpha,
            reference_method=args.reference_method)
        pairwise_df.to_csv(args.output_dir / "pairwise_wilcoxon_holm.csv", index=False)
    else:
        print("  [Wilcoxon] desabilitado. Use --wilcoxon para ativar.")

    nem, status = try_nemenyi(wide, metric, args.output_dir)
    print(f"  {status}")

    if args.latex:
        save_latex_tables(friedman_df, rank_df, pairwise_df,
                          args.output_dir, label_prefix=LABEL)

    print("\n  Gerando gráficos...")
    n_ds = int(friedman_row["N_Datasets"])
    plot_cd_diagram(rank_df, metric, n_ds, args.alpha, nem, args.output_dir)
    if nem is not None:
        plot_nemenyi_heatmap(nem, metric, args.alpha, args.output_dir)
    plot_win_counts_mse(wins_count_df, ALL_METHODS, args.output_dir)

    print(f"\n  Resultados em: {args.output_dir}")


if __name__ == "__main__":
    main()
