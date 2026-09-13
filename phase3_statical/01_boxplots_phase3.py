# -*- coding: utf-8 -*-
"""
Script 01 — Boxplots dos resultados da Fase 3 do DREAM
======================================================

Objetivo
--------
Gerar boxplots comparativos por método e métrica a partir do arquivo
dream_phase3_results.csv.

Entradas esperadas
------------------
CSV com, pelo menos, as colunas:
Dataset, Exec, Fold, Method, MSE, MAE, RMSE, MAPE, R2, MedAE

Uso básico
----------
python 01_boxplots_phase3.py --input dream_phase3_results.csv

Uso recomendado para tese
-------------------------
python 01_boxplots_phase3.py \
    --input dream_phase3_results.csv \
    --output-dir outputs_boxplots \
    --metrics MSE RMSE MAE MedAE R2 \
    --level dataset \
    --exclude-oracles

Observação metodológica
-----------------------
Por padrão, o script usa level="dataset", isto é, calcula a média por
Dataset x Method antes do boxplot. Essa opção evita que execuções/folds sejam
tratados como amostras independentes na comparação global entre datasets.

Para visualizar a dispersão interna por execução/fold, use:
python 01_boxplots_phase3.py --level fold
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuração tipográfica global — compatível com Times New Roman (tese ABNT)
# matplotlib.rcParams aplicado uma vez; afeta todas as figuras do script.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    # Família de fonte serifada equivalente a Times New Roman
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    # Tamanhos — corpo 12pt da tese; legenda/tick levemente menores
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    # Renderização vetorial para evitar artefatos em alta resolução
    "figure.dpi":         150,   # pré-visualização; salvamento usa DPI_EXPORT
    "savefig.dpi":        600,   # resolução de exportação para a tese
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    # Grade discreta e fundo branco
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    # Bordas do gráfico
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Dimensões da figura em polegadas para página A4 com margens ABNT
# Largura útil ≈ 15 cm (3 cm esquerda + 2 cm direita) → 5.91 pol.
# Figuras de coluna única com 3 métodos: 5.91 pol. × proporção 4:3
FIGURE_WIDTH_IN  = 5.91   # largura única coluna — página inteira é 5.91 pol.
FIGURE_HEIGHT_IN = 4.13   # razão áurea aproximada (~1.43)

# Rótulos de exibição: substitui underscore por hífen nos eixos
METHOD_DISPLAY_LABELS: dict[str, str] = {
    "DREAM_DS":  "DREAM-DS",
    "DREAM_DW":  "DREAM-DW",
    "DREAM_DWS": "DREAM-DWS",
}

def method_label(method: str) -> str:
    """Retorna rótulo de exibição do método (ex.: DREAM_DS → DREAM-DS)."""
    return METHOD_DISPLAY_LABELS.get(method, method.replace("_", "-"))


DEFAULT_METRICS = ["MSE", "RMSE", "MAE", "MedAE", "R2"]
DEFAULT_ORACLE_METHODS = {"ORACLE_BASE", "KMEN_ORACLE"}

# ---------------------------------------------------------------------------
# Conjunto de métodos avaliados por etapa experimental.
# Altere ACTIVE_METHODS para expandir a comparação:
#   Etapa 1 — estratégias DREAM entre si:
#       ["DREAM_DS", "DREAM_DW", "DREAM_DWS"]
#   Etapa 2 — incluir baselines estáticos:
#       ["DREAM_DS", "DREAM_DW", "DREAM_DWS", "TOP3_AVG", "TOP5_AVG", "TOP_ALL_AVG", "MINE_DS", "MINE_DW", "MINE_DWS"]
#   Etapa 3 — incluir oráculos:
#       ["DREAM_DS", "DREAM_DW", "DREAM_DWS", ..., "ORACLE_BASE", "KMEN_ORACLE"]
# ---------------------------------------------------------------------------
ACTIVE_METHODS: list[str] | None = ["DREAM_DS", "DREAM_DW", "DREAM_DWS"]


def sanitize_filename(value: str) -> str:
    """Converte texto para nome seguro de arquivo."""
    value = str(value)
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Valida se o CSV contém as colunas necessárias."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "O arquivo de entrada não contém as colunas obrigatórias: "
            + ", ".join(missing)
        )


def resolve_methods(
    df: pd.DataFrame,
    methods: list[str] | None,
    exclude_oracles: bool,
) -> list[str]:
    """Define a lista de métodos a serem plotados preservando a ordem do CSV."""
    available = list(df["Method"].dropna().astype(str).unique())

    if methods:
        missing = [m for m in methods if m not in available]
        if missing:
            raise ValueError(
                "Os seguintes métodos não foram encontrados no CSV: "
                + ", ".join(missing)
            )
        selected = methods
    else:
        selected = available

    if exclude_oracles:
        selected = [m for m in selected if m not in DEFAULT_ORACLE_METHODS]

    if not selected:
        raise ValueError("Nenhum método disponível após os filtros aplicados.")

    return selected


def prepare_data(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    level: str,
) -> pd.DataFrame:
    """
    Prepara os dados para boxplot.

    level="dataset":
        uma observação por Dataset x Method, obtida pela média sobre Exec/Fold.
        É a opção recomendada para comparação global entre datasets.

    level="fold":
        uma observação por Dataset x Exec x Fold x Method.
        Útil para análise exploratória da dispersão interna.
    """
    cols_needed = ["Dataset", "Exec", "Fold", "Method", metric]
    validate_columns(df, cols_needed)

    data = df.loc[df["Method"].isin(methods), cols_needed].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric, "Dataset", "Method"])

    if level == "dataset":
        data = (
            data.groupby(["Dataset", "Method"], as_index=False)[metric]
            .mean()
            .sort_values(["Dataset", "Method"])
        )
    elif level == "fold":
        data = (
            data.groupby(["Dataset", "Exec", "Fold", "Method"], as_index=False)[metric]
            .mean()
            .sort_values(["Dataset", "Exec", "Fold", "Method"])
        )
    else:
        raise ValueError("level deve ser 'dataset' ou 'fold'.")

    return data


def plot_global_boxplot(
    data: pd.DataFrame,
    metric: str,
    methods: list[str],
    output_path: Path,
    level: str,
    log_scale_errors: bool = False,
) -> None:
    """
    Gera um boxplot global por método para uma métrica.

    Configurações tipográficas:
    - Fonte serifada (Times New Roman) compatível com a tese ABNT/PPGEc-UPE
    - Dimensões em polegadas para página A4 com margens ABNT (largura útil 15 cm)
    - Exportação a 600 dpi (definido via mpl.rcParams["savefig.dpi"])
    - Rótulos dos métodos com hífen (ex.: DREAM-DS) para publicação

    Linhas do boxplot:
    - Laranja sólida: mediana
    - Verde tracejada: média
    """
    values = [
        data.loc[data["Method"] == method, metric].dropna().to_numpy()
        for method in methods
    ]

    non_empty = [v for v in values if len(v) > 0]
    if not non_empty:
        raise ValueError(f"Não há dados válidos para a métrica {metric}.")

    display_labels = [method_label(m) for m in methods]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    bp = ax.boxplot(
        values,
        tick_labels=display_labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        widths=0.45,
        boxprops=dict(facecolor="white", color="black", linewidth=0.8),
        medianprops=dict(color="#E07B00", linewidth=1.5),
        meanprops=dict(color="#2A7A2A", linewidth=1.2, linestyle="--"),
        whiskerprops=dict(color="black", linewidth=0.8, linestyle="-"),
        capprops=dict(color="black", linewidth=0.8),
        flierprops=dict(
            marker="o", markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=0.7, markersize=4,
        ),
    )

    ax.set_xlabel("Método")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=0)

    # Legenda das linhas internas
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#E07B00", linewidth=1.5, label="Mediana"),
        Line2D([0], [0], color="#2A7A2A", linewidth=1.2, linestyle="--", label="Média"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9)

    if log_scale_errors and metric.upper() != "R2":
        positive_values = np.concatenate([v[v > 0] for v in non_empty if np.any(v > 0)])
        if len(positive_values) > 0:
            ax.set_yscale("log")
            ax.set_ylabel(f"{metric} (escala log)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_boxplots_by_dataset(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    output_dir: Path,
    max_cols: int = 3,
) -> None:
    """
    Gera boxplots por dataset em arquivos separados.

    Cada figura apresenta, para um dataset, a distribuição de Exec/Fold
    por método. Essa saída é útil para analisar comportamento local por base.
    """
    data = prepare_data(df, metric=metric, methods=methods, level="fold")

    for dataset, subset in data.groupby("Dataset"):
        values = [
            subset.loc[subset["Method"] == method, metric].dropna().to_numpy()
            for method in methods
        ]
        if all(len(v) == 0 for v in values):
            continue

        display_labels = [method_label(m) for m in methods]

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

        ax.boxplot(
            values,
            tick_labels=display_labels,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            widths=0.45,
            boxprops=dict(facecolor="white", color="black", linewidth=0.8),
            medianprops=dict(color="#E07B00", linewidth=1.5),
            meanprops=dict(color="#2A7A2A", linewidth=1.2, linestyle="--"),
            whiskerprops=dict(color="black", linewidth=0.8, linestyle="-"),
            capprops=dict(color="black", linewidth=0.8),
            flierprops=dict(
                marker="o", markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=0.7, markersize=4,
            ),
        )

        ax.set_xlabel("Método")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        output_path = output_dir / f"boxplot_{sanitize_filename(dataset)}_{metric}.png"
        fig.savefig(output_path)
        plt.close(fig)


def save_descriptive_summary(
    df: pd.DataFrame,
    metrics: list[str],
    methods: list[str],
    level: str,
    output_dir: Path,
) -> None:
    """Salva estatísticas descritivas usadas nos boxplots."""
    rows = []

    for metric in metrics:
        data = prepare_data(df, metric=metric, methods=methods, level=level)
        for method, subset in data.groupby("Method"):
            values = subset[metric].dropna()
            rows.append({
                "Metric": metric,
                "Method": method,
                "N": int(values.shape[0]),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Std": float(values.std(ddof=1)) if values.shape[0] > 1 else np.nan,
                "Min": float(values.min()),
                "Q1": float(values.quantile(0.25)),
                "Q3": float(values.quantile(0.75)),
                "Max": float(values.max()),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "boxplot_descriptive_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera boxplots comparativos dos resultados da Fase 3 do DREAM."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase3" / "dream_phase3_results.csv",
        help="Caminho para o CSV de resultados da Fase 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase3" / "dream_boxplots",
        help="Diretório de saída das figuras e resumos.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Métricas a serem plotadas. Ex.: MSE RMSE MAE MedAE R2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ACTIVE_METHODS,
        help=(
            "Lista de métodos a plotar. "
            "O padrão é definido pela constante ACTIVE_METHODS no topo do script. "
            "Passe None para usar todos os métodos disponíveis no CSV."
        ),
    )
    parser.add_argument(
        "--level",
        choices=["dataset", "fold"],
        default="dataset",
        help="Nível de agregação: 'dataset' ou 'fold'.",
    )
    parser.add_argument(
        "--exclude-oracles",
        action="store_true",
        help="Remove ORACLE_BASE e KMEN_ORACLE dos gráficos.",
    )
    parser.add_argument(
        "--include-dataset-plots",
        action="store_true",
        help="Também gera boxplots separados por dataset usando Exec/Fold.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help=(
            "Resolução de exportação das figuras em DPI. "
            "Padrão: 600 (tese/impressão). Use 150 para provas rápidas."
        ),
    )
    parser.add_argument(
        "--log-scale-errors",
        action="store_true",
        default=False,
        help=(
            "Usa escala logarítmica para métricas de erro nos boxplots globais. "
            "Desabilitado por padrão. Use apenas para análise exploratória; "
            "não recomendado para figuras da tese."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    # Aplica DPI de exportação conforme argumento (sobrescreve o padrão do rcParams)
    mpl.rcParams["savefig.dpi"] = args.dpi

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validate_columns(df, ["Dataset", "Exec", "Fold", "Method"])

    metrics = [m for m in args.metrics if m in df.columns]
    missing_metrics = [m for m in args.metrics if m not in df.columns]
    if missing_metrics:
        print(
            "[AVISO] Métricas ignoradas por não existirem no CSV: "
            + ", ".join(missing_metrics)
        )

    if not metrics:
        raise ValueError("Nenhuma métrica válida foi encontrada no CSV.")

    methods = resolve_methods(
        df=df,
        methods=args.methods,
        exclude_oracles=args.exclude_oracles,
    )

    print("Métodos usados:", ", ".join(methods))
    print("Métricas usadas:", ", ".join(metrics))
    print("Nível de agregação:", args.level)

    for metric in metrics:
        data = prepare_data(df, metric=metric, methods=methods, level=args.level)
        output_path = args.output_dir / f"boxplot_global_{metric}_{args.level}.png"

        plot_global_boxplot(
            data=data,
            metric=metric,
            methods=methods,
            output_path=output_path,
            level=args.level,
            log_scale_errors=args.log_scale_errors,
        )

        print(f"[OK] Figura salva: {output_path}")

        if args.include_dataset_plots:
            dataset_dir = args.output_dir / f"por_dataset_{metric}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            plot_boxplots_by_dataset(
                df=df,
                metric=metric,
                methods=methods,
                output_dir=dataset_dir,
            )
            print(f"[OK] Figuras por dataset salvas em: {dataset_dir}")

    save_descriptive_summary(
        df=df,
        metrics=metrics,
        methods=methods,
        level=args.level,
        output_dir=args.output_dir,
    )
    print(f"[OK] Resumo descritivo salvo em: {args.output_dir / 'boxplot_descriptive_summary.csv'}")


if __name__ == "__main__":
    main()