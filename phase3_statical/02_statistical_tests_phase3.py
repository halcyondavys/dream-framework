# -*- coding: utf-8 -*-
"""
Script 02 — Testes estatísticos dos resultados da Fase 3 do DREAM
=================================================================

Objetivo
--------
Executar testes estatísticos comparativos entre métodos a partir do arquivo
dream_phase3_results.csv.

Testes implementados
--------------------
1. Friedman por métrica, usando Dataset como bloco experimental.
2. Ranking médio dos métodos por métrica.
3. Wilcoxon pareado entre pares de métodos, com correção de Holm (opcional, --wilcoxon).
4. Nemenyi pós-Friedman, quando a biblioteca scikit-posthocs estiver instalada.
5. Contagem de vitórias/derrotas/empates por dataset.

Gráficos gerados
----------------
- CD diagram (Critical Difference) por métrica
- Heatmap da matriz Nemenyi por métrica
- Bar chart de contagem de vitórias por método e métrica

Uso recomendado para tese
-------------------------
python 02_statistical_tests_phase3.py \
    --output-dir outputs_stats \
    --metrics MSE RMSE MAE MedAE R2 \
    --exclude-oracles \
    --reference-method DREAM_DW \
    --latex

Dependências principais
-----------------------
pip install pandas numpy scipy matplotlib

Dependência opcional para Nemenyi:
pip install scikit-posthocs
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Configuração tipográfica global — compatível com Times New Roman (tese ABNT)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Dimensões padrão para página A4 com margens ABNT (largura útil ≈ 15 cm = 5.91 pol.)
FIGURE_WIDTH_IN  = 5.91
FIGURE_HEIGHT_IN = 4.13

# Rótulos de exibição dos métodos
METHOD_DISPLAY_LABELS: dict[str, str] = {
    "DREAM_DS":   "DREAM-DS",
    "DREAM_DW":   "DREAM-DW",
    "DREAM_DWS":  "DREAM-DWS",
}

def method_label(method: str) -> str:
    return METHOD_DISPLAY_LABELS.get(method, method.replace("_", "-"))


# ---------------------------------------------------------------------------
# Constantes e configuração de métodos ativos
# ---------------------------------------------------------------------------
DEFAULT_METRICS = ["MSE", "RMSE", "MAE", "MedAE", "R2"]
DEFAULT_ORACLE_METHODS = {"ORACLE_BASE", "KMEN_ORACLE"}
HIGHER_IS_BETTER = {"R2"}

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


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "O arquivo de entrada não contém as colunas obrigatórias: "
            + ", ".join(missing)
        )


def sanitize_filename(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def resolve_methods(
    df: pd.DataFrame,
    methods: list[str] | None,
    exclude_oracles: bool,
) -> list[str]:
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

    if len(selected) < 2:
        raise ValueError(
            "São necessários ao menos dois métodos para executar testes estatísticos."
        )

    return selected


def metric_direction(metric: str) -> str:
    """Retorna 'max' para métricas em que maior é melhor; caso contrário, 'min'."""
    return "max" if metric.upper() in HIGHER_IS_BETTER else "min"


# ---------------------------------------------------------------------------
# Agregação e ranking
# ---------------------------------------------------------------------------
def aggregate_by_dataset(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
) -> pd.DataFrame:
    """
    Agrega por Dataset x Method.

    Retorna uma matriz wide:
    linhas = datasets
    colunas = métodos
    valores = média da métrica em Exec/Fold.
    """
    validate_columns(df, ["Dataset", "Exec", "Fold", "Method", metric])

    data = df.loc[df["Method"].isin(methods), ["Dataset", "Exec", "Fold", "Method", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric, "Dataset", "Method"])

    grouped = data.groupby(["Dataset", "Method"], as_index=False)[metric].mean()
    wide = grouped.pivot(index="Dataset", columns="Method", values=metric)
    wide = wide.dropna(axis=0, how="any")

    methods_present = [m for m in methods if m in wide.columns]
    wide = wide[methods_present]

    if wide.shape[0] < 2:
        raise ValueError(
            f"A métrica {metric} possui menos de 2 datasets completos após filtros."
        )
    if wide.shape[1] < 2:
        raise ValueError(
            f"A métrica {metric} possui menos de 2 métodos completos após filtros."
        )

    return wide


def rank_methods(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Calcula rankings por dataset e ranking médio por método.

    Para métricas de erro: menor valor = melhor ranking.
    Para R2: maior valor = melhor ranking.
    """
    ascending = metric_direction(metric) == "min"
    ranks = wide.rank(axis=1, method="average", ascending=ascending)

    rank_summary = pd.DataFrame({
        "Metric": metric,
        "Method": ranks.columns,
        "Average_Rank": ranks.mean(axis=0).values,
        "Median_Rank": ranks.median(axis=0).values,
        "Std_Rank": ranks.std(axis=0, ddof=1).values,
        "N_Datasets": wide.shape[0],
    }).sort_values(["Average_Rank", "Method"])

    return rank_summary


# ---------------------------------------------------------------------------
# Testes estatísticos
# ---------------------------------------------------------------------------
def friedman_test(wide: pd.DataFrame, metric: str) -> dict:
    """Executa Friedman considerando datasets como blocos e métodos como tratamentos."""
    arrays = [wide[col].to_numpy() for col in wide.columns]
    stat, p_value = friedmanchisquare(*arrays)

    return {
        "Metric": metric,
        "N_Datasets": int(wide.shape[0]),
        "N_Methods": int(wide.shape[1]),
        "Friedman_Statistic": float(stat),
        "P_Value": float(p_value),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Correção de Holm-Bonferroni sem dependência externa."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        raw_p = p_values[idx]
        adj_p = (m - rank) * raw_p
        running_max = max(running_max, adj_p)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def paired_wilcoxon_all_pairs(
    wide: pd.DataFrame,
    metric: str,
    alpha: float,
    reference_method: str | None = None,
) -> pd.DataFrame:
    """
    Executa Wilcoxon pareado para todos os pares de métodos.
    Ativado apenas com --wilcoxon.
    """
    direction = metric_direction(metric)
    rows = []

    for method_a, method_b in itertools.combinations(wide.columns, 2):
        a = wide[method_a].astype(float)
        b = wide[method_b].astype(float)
        diff = a - b

        if np.allclose(diff.to_numpy(), 0.0):
            stat = 0.0
            p_value = 1.0
        else:
            try:
                stat, p_value = wilcoxon(
                    a, b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    mode="auto",
                )
            except ValueError:
                stat = np.nan
                p_value = np.nan

        if direction == "min":
            wins_a = int((a < b).sum())
            wins_b = int((b < a).sum())
            better_mean = method_a if a.mean() < b.mean() else method_b
        else:
            wins_a = int((a > b).sum())
            wins_b = int((b > a).sum())
            better_mean = method_a if a.mean() > b.mean() else method_b

        ties = int((np.isclose(a, b)).sum())

        rows.append({
            "Metric": metric,
            "Method_A": method_a,
            "Method_B": method_b,
            "Reference_Comparison": bool(
                reference_method and
                (method_a == reference_method or method_b == reference_method)
            ),
            "N_Datasets": int(wide.shape[0]),
            "Wilcoxon_Statistic": float(stat) if not pd.isna(stat) else np.nan,
            "P_Value": float(p_value) if not pd.isna(p_value) else np.nan,
            "Mean_A": float(a.mean()),
            "Mean_B": float(b.mean()),
            "Median_A": float(a.median()),
            "Median_B": float(b.median()),
            "Mean_Diff_A_minus_B": float(diff.mean()),
            "Median_Diff_A_minus_B": float(diff.median()),
            "Wins_A": wins_a,
            "Wins_B": wins_b,
            "Ties": ties,
            "Better_By_Mean": better_mean,
            "Direction": direction,
        })

    result = pd.DataFrame(rows)

    valid_mask = result["P_Value"].notna()
    adjusted = [np.nan] * len(result)
    if valid_mask.any():
        adjusted_valid = holm_adjust(result.loc[valid_mask, "P_Value"].tolist())
        valid_indices = result.index[valid_mask].tolist()
        for idx, adj_p in zip(valid_indices, adjusted_valid):
            adjusted[idx] = adj_p

    result["P_Adjusted_Holm"] = adjusted
    result["Significant_Holm"] = result["P_Adjusted_Holm"] < alpha

    return result.sort_values(["Metric", "P_Adjusted_Holm", "P_Value", "Method_A", "Method_B"])


def wins_by_dataset(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Identifica o melhor método em cada dataset e contabiliza vitórias."""
    direction = metric_direction(metric)

    if direction == "min":
        best_method = wide.idxmin(axis=1)
        best_value  = wide.min(axis=1)
    else:
        best_method = wide.idxmax(axis=1)
        best_value  = wide.max(axis=1)

    rows = []
    for dataset in wide.index:
        rows.append({
            "Metric":      metric,
            "Dataset":     dataset,
            "Best_Method": best_method.loc[dataset],
            "Best_Value":  float(best_value.loc[dataset]),
            "Direction":   direction,
        })

    return pd.DataFrame(rows)


def try_nemenyi(
    wide: pd.DataFrame,
    metric: str,
    output_dir: Path,
    methods: list[str],
    alpha: float,
) -> tuple[pd.DataFrame | None, str]:
    """
    Executa Nemenyi se scikit-posthocs estiver instalado.

    Retorna (matriz_nemenyi, mensagem_status).
    Retorna (None, mensagem) se scikit-posthocs não estiver disponível.
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        return None, (
            "Nemenyi não executado porque scikit-posthocs não está instalado. "
            "Instale com: pip install scikit-posthocs"
        )

    data = wide.copy()
    if metric_direction(metric) == "max":
        data = -data

    nemenyi = sp.posthoc_nemenyi_friedman(data.to_numpy())
    nemenyi.index   = wide.columns
    nemenyi.columns = wide.columns

    output_path = output_dir / f"nemenyi_{sanitize_filename(metric)}.csv"
    nemenyi.to_csv(output_path)

    return nemenyi, f"Nemenyi salvo em: {output_path}"


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _nemenyi_cd_value(n_methods: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Calcula o valor crítico de diferença (CD) para o teste de Nemenyi.

    Fórmula de Demšar (2006):
        CD = q_alpha * sqrt(k(k+1) / (6*N))

    q_alpha para alpha=0.05 (tabela de Demšar, distribuição studentized range):
        k=2: 1.960, k=3: 2.343, k=4: 2.569, k=5: 2.728, k=6: 2.850,
        k=7: 2.949, k=8: 3.031, k=9: 3.102, k=10: 3.164
    """
    if n_datasets < 2:
        raise ValueError(
            "O cálculo da diferença crítica exige pelo menos 2 datasets."
        )

    if not np.isclose(alpha, 0.05):
        raise ValueError(
            "O cálculo de CD implementado possui valores críticos "
            "apenas para alpha = 0.05."
        )

    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }

    if n_methods not in q_alpha_table:
        raise ValueError(
            "O cálculo de CD está configurado apenas para "
            "comparações entre 2 e 10 métodos."
        )

    q_alpha = q_alpha_table[n_methods]

    cd = q_alpha * np.sqrt(
        n_methods * (n_methods + 1) / (6 * n_datasets)
    )

    return cd


def plot_cd_diagram(
    rank_df: pd.DataFrame,
    metric: str,
    n_datasets: int,
    alpha: float,
    nemenyi_matrix: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """
    Gera o diagrama de Diferença Crítica (CD diagram) para uma métrica.

    Exibe o ranking médio dos métodos em uma linha horizontal.
    Métodos sem diferença estatisticamente significativa (p > alpha no Nemenyi
    ou distância ≤ CD) são conectados por uma barra horizontal grossa.

    Referência: Demšar (2006), J. Machine Learning Research.
    """
    subset = rank_df[rank_df["Metric"] == metric].sort_values("Average_Rank")
    methods  = subset["Method"].tolist()
    ranks    = subset["Average_Rank"].tolist()
    n_methods = len(methods)

    if n_methods < 2:
        return

    cd = _nemenyi_cd_value(n_methods, n_datasets, alpha)

    # Determina grupos não-significativos
    # Usa matriz Nemenyi se disponível; caso contrário, usa CD diretamente.
    def _not_significant(m1: str, m2: str) -> bool:
        if nemenyi_matrix is not None and m1 in nemenyi_matrix.index and m2 in nemenyi_matrix.columns:
            return float(nemenyi_matrix.loc[m1, m2]) > alpha
        r1 = ranks[methods.index(m1)]
        r2 = ranks[methods.index(m2)]
        return abs(r1 - r2) <= cd

    # Determina os grupos máximos de métodos sem diferença significativa.
    #
    # A ausência de significância não é uma relação transitiva:
    # A pode não diferir de B, e B pode não diferir de C,
    # enquanto A difere significativamente de C.
    #
    # Portanto, não se deve utilizar Union-Find. Cada grupo apresentado
    # no diagrama deve formar uma clique, isto é, todos os pares internos
    # devem apresentar p > alpha no pós-teste de Nemenyi.

    candidate_cliques: list[tuple[str, ...]] = []

    for group_size in range(2, len(methods) + 1):
        for candidate in itertools.combinations(methods, group_size):
            all_pairs_not_significant = all(
                _not_significant(method_a, method_b)
                for method_a, method_b in itertools.combinations(candidate, 2)
            )

            if all_pairs_not_significant:
                candidate_cliques.append(candidate)

    # Mantém apenas cliques máximas. Uma clique é descartada quando está
    # integralmente contida em outra clique válida de maior tamanho.
    cliques: list[list[str]] = []

    for candidate in candidate_cliques:
        candidate_set = set(candidate)

        is_maximal = not any(
            candidate_set < set(other)
            for other in candidate_cliques
        )

        if is_maximal:
            ordered_clique = [method for method in methods if method in candidate_set]
            cliques.append(ordered_clique)

    # Remove eventuais duplicações, preservando a ordem dos ranks.
    unique_cliques: list[list[str]] = []
    seen_cliques: set[tuple[str, ...]] = set()

    for clique in cliques:
        clique_key = tuple(clique)

        if clique_key not in seen_cliques:
            seen_cliques.add(clique_key)
            unique_cliques.append(clique)

    cliques = unique_cliques

    # Layout da figura
    fig_h = max(FIGURE_HEIGHT_IN, 0.5 * n_methods + 1.5)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, fig_h))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-0.5, n_methods + 0.5)
    ax.axis("off")

    rank_min = min(ranks)
    rank_max = max(ranks)
    rank_range = max(rank_max - rank_min, 1.0)

    def _rank_to_x(r: float) -> float:
        """Mapeia ranking para coordenada X normalizada."""
        return 1.0 + (r - rank_min) / rank_range * (n_methods - 1)

    # Linha de eixo central
    y_axis = n_methods * 0.5
    ax.axhline(y=y_axis, xmin=0.05, xmax=0.95, color="black", linewidth=1.2, zorder=1)

    # Barra de CD — posicionada acima da região dos rótulos superiores
    x_cd_start = _rank_to_x(rank_min)
    x_cd_end   = _rank_to_x(rank_min + cd)
    y_cd = n_methods * 0.94
    ax.annotate(
        "", xy=(x_cd_end, y_cd), xytext=(x_cd_start, y_cd),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    ax.text(
        (x_cd_start + x_cd_end) / 2, y_cd + 0.28,
        f"CD = {cd:.3f}",
        ha="center", va="bottom", fontsize=9,
    )

    # Posições verticais dos métodos (acima e abaixo da linha)
    top_methods    = methods[: n_methods // 2 + n_methods % 2]
    bottom_methods = methods[n_methods // 2 + n_methods % 2 :]

    y_top_base    = y_axis + 0.6
    y_bottom_base = y_axis - 0.6
    y_step = 0.55

    method_y: dict[str, float] = {}
    for i, m in enumerate(top_methods):
        method_y[m] = y_top_base + i * y_step
    for i, m in enumerate(bottom_methods):
        method_y[m] = y_bottom_base - i * y_step

    for m, r in zip(methods, ranks):
        x = _rank_to_x(r)
        y = method_y[m]
        is_top = m in top_methods

        # Linha vertical do método até a linha de eixo
        ax.plot([x, x], [y_axis, y], color="black", linewidth=0.8, zorder=2)
        # Ponto no eixo
        ax.plot(x, y_axis, "o", color="black", markersize=4, zorder=3)

        label = f"{method_label(m)}\n(rank {r:.2f})"
        va = "bottom" if is_top else "top"
        ax.text(
            x, y + (0.05 if is_top else -0.05),
            label, ha="center", va=va, fontsize=9,
        )

    # Barras de não-significância (cliques)
    clique_colors = ["#2A6FAC", "#C0392B", "#27AE60", "#8E44AD", "#E67E22"]
    y_clique_base = y_axis - 0.22
    for ci, clique in enumerate(cliques):
        clique_ranks = [ranks[methods.index(m)] for m in clique]
        x_left  = _rank_to_x(min(clique_ranks)) - 0.02
        x_right = _rank_to_x(max(clique_ranks)) + 0.02
        y_c = y_clique_base - ci * 0.12
        color = clique_colors[ci % len(clique_colors)]
        ax.plot(
            [x_left, x_right], [y_c, y_c],
            color=color, linewidth=3.5, solid_capstyle="round", zorder=4,
            label=f"Grupo {ci+1} (p > {alpha})",
        )

    if cliques:
        ax.legend(
            loc="lower center", fontsize=8, frameon=True, framealpha=0.9,
            ncol=min(len(cliques), 3),
        )

    ax.set_title(f"CD Diagram — {metric} (α = {alpha})", pad=6)

    fig.tight_layout()
    out = output_dir / f"cd_diagram_{sanitize_filename(metric)}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] CD diagram salvo: {out}")


def plot_nemenyi_heatmap(
    nemenyi_matrix: pd.DataFrame,
    metric: str,
    alpha: float,
    output_dir: Path,
) -> None:
    """
    Gera heatmap da matriz de p-valores do teste de Nemenyi.

    Células com p > alpha (diferença não-significativa) são destacadas
    em azul claro; células significativas em laranja/vermelho.
    """
    methods = list(nemenyi_matrix.columns)
    n = len(methods)
    labels = [method_label(m) for m in methods]

    matrix = nemenyi_matrix.to_numpy(dtype=float)

    fig_size = max(FIGURE_WIDTH_IN, 0.9 * n + 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Colormap: valores acima de alpha (não-sig.) em azul claro; abaixo em vermelho
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nemenyi",
        [(0.0, "#C0392B"), (alpha, "#F7DC6F"), (alpha + 0.001, "#AED6F1"), (1.0, "#2874A6")],
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Anotações das células
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if i == j:
                text = "—"
                color = "black"
            else:
                text = f"{val:.3f}"
                color = "white" if val < alpha * 0.5 or val > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p-valor (Nemenyi)", fontsize=9)
    cbar.ax.axhline(alpha, color="black", linewidth=1.2, linestyle="--")
    cbar.ax.text(1.5, alpha, f" α = {alpha}", va="center", fontsize=8)

    ax.set_title(f"Nemenyi post-hoc — {metric}", pad=8)
    ax.spines[:].set_visible(False)
    ax.grid(False)

    fig.tight_layout()
    out = output_dir / f"nemenyi_heatmap_{sanitize_filename(metric)}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Heatmap Nemenyi salvo: {out}")


def plot_win_counts(
    wins_count_df: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
) -> None:
    """
    Gera bar chart de contagem de vitórias por método para cada métrica.

    Uma figura por métrica; métodos ordenados por número de vitórias (decrescente).
    """
    for metric in metrics:
        subset = wins_count_df[wins_count_df["Metric"] == metric].copy()
        if subset.empty:
            continue

        subset = subset.sort_values("Wins", ascending=False)
        display_labels = [method_label(m) for m in subset["Best_Method"]]
        wins = subset["Wins"].tolist()
        n_datasets = wins_count_df[wins_count_df["Metric"] == metric]["Wins"].sum()

        colors = ["#2A6FAC", "#E07B00", "#27AE60", "#8E44AD", "#C0392B",
                  "#16A085", "#D35400", "#2C3E50"]

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

        bars = ax.bar(
            display_labels, wins,
            color=colors[: len(display_labels)],
            width=0.55,
            edgecolor="black",
            linewidth=0.6,
        )

        # Rótulo de valor sobre cada barra
        for bar, w in zip(bars, wins):
            pct = w / n_datasets * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{w}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xlabel("Método")
        ax.set_ylabel("Nº de vitórias")
        ax.set_ylim(0, n_datasets + 2)
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        out = output_dir / f"win_counts_{sanitize_filename(metric)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  [OK] Bar chart de vitórias salvo: {out}")


# ---------------------------------------------------------------------------
# Tabelas LaTeX
# ---------------------------------------------------------------------------
def save_latex_tables(
    friedman_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    pairwise_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """
    Gera versões LaTeX das tabelas principais com formatação booktabs.

    - booktabs (\\toprule / \\midrule / \\bottomrule)
    - \\fonte{Elaborado pelo autor.} após cada tabela
    - Valores numéricos com 4 casas decimais
    """

    def _booktabs_table(
        df: pd.DataFrame,
        caption: str,
        label: str,
        float_fmt: str = "{:.4f}",
    ) -> str:
        cols = list(df.columns)
        col_spec = "l" + "r" * (len(cols) - 1)

        def fmt_cell(v: object) -> str:
            if isinstance(v, float):
                if pd.isna(v):
                    return "--"
                return float_fmt.format(v)
            if isinstance(v, bool):
                return "Sim" if v else "Não"
            return str(v)

        header = " & ".join(cols) + r" \\"
        rows_tex = []
        for _, row in df.iterrows():
            cells = " & ".join(fmt_cell(row[c]) for c in cols)
            rows_tex.append(cells + r" \\")

        body = "\n    ".join(rows_tex)

        return (
            "\\begin{table}[htbp]\n"
            "  \\centering\n"
            f"  \\caption{{{caption}}}\n"
            f"  \\label{{{label}}}\n"
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            "    \\toprule\n"
            f"    {header}\n"
            "    \\midrule\n"
            f"    {body}\n"
            "    \\bottomrule\n"
            "  \\end{tabular}\n"
            "  \\fonte{Elaborado pelo autor.}\n"
            "\\end{table}\n"
        )

    with open(output_dir / "friedman_summary.tex", "w", encoding="utf-8") as f:
        f.write(_booktabs_table(
            friedman_df,
            caption="Resultados do teste de Friedman por métrica — Fase 3 do framework DREAM.",
            label="tab:friedman_phase3",
        ))

    with open(output_dir / "average_ranks.tex", "w", encoding="utf-8") as f:
        f.write(_booktabs_table(
            rank_df,
            caption="Ranking médio dos métodos por métrica (protocolo Demšar).",
            label="tab:average_ranks_phase3",
        ))

    if pairwise_df is not None:
        cols_reduced = [
            "Metric", "Method_A", "Method_B", "P_Value", "P_Adjusted_Holm",
            "Significant_Holm", "Wins_A", "Wins_B", "Ties", "Better_By_Mean",
        ]
        available_cols = [c for c in cols_reduced if c in pairwise_df.columns]
        with open(output_dir / "pairwise_wilcoxon_holm_reduced.tex", "w", encoding="utf-8") as f:
            f.write(_booktabs_table(
                pairwise_df[available_cols],
                caption=(
                    "Teste de Wilcoxon pareado com correção de Holm — "
                    "comparações pairwise entre métodos da Fase 3."
                ),
                label="tab:wilcoxon_holm_phase3",
            ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa Friedman, Nemenyi e gera gráficos para os resultados da Fase 3."
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
        default=PROJECT_ROOT / "results" / "phase3" / "dream_statistic",
        help="Diretório de saída dos resultados estatísticos e gráficos.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Métricas avaliadas. Ex.: MSE RMSE MAE MedAE R2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=ACTIVE_METHODS,
        help=(
            "Lista de métodos a comparar. "
            "O padrão é definido pela constante ACTIVE_METHODS no topo do script."
        ),
    )
    parser.add_argument(
        "--exclude-oracles",
        action="store_true",
        help="Remove ORACLE_BASE e KMEN_ORACLE dos testes principais.",
    )
    parser.add_argument(
        "--reference-method",
        type=str,
        default=None,
        help="Método de referência para marcar comparações pareadas. Ex.: DREAM_DW",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância.",
    )
    parser.add_argument(
        "--wilcoxon",
        action="store_true",
        default=False,
        help=(
            "Executa Wilcoxon pareado com correção de Holm para todos os pares. "
            "Desabilitado por padrão. Use apenas para análises pairwise pontuais (QP2/QP3)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolução de exportação dos gráficos em DPI. Padrão: 600.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Também salva tabelas LaTeX com formatação booktabs.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    mpl.rcParams["savefig.dpi"] = args.dpi

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    validate_columns(df, ["Dataset", "Exec", "Fold", "Method"])

    metrics = [m for m in args.metrics if m in df.columns]
    missing_metrics = [m for m in args.metrics if m not in df.columns]
    if missing_metrics:
        print("[AVISO] Métricas ignoradas por não existirem no CSV: " + ", ".join(missing_metrics))

    if not metrics:
        raise ValueError("Nenhuma métrica válida foi encontrada no CSV.")

    methods = resolve_methods(df=df, methods=args.methods, exclude_oracles=args.exclude_oracles)

    if args.reference_method and args.reference_method not in methods:
        print(f"[AVISO] O método de referência '{args.reference_method}' não está na lista de métodos.")

    print("Métodos testados:", ", ".join(methods))
    print("Métricas testadas:", ", ".join(metrics))
    print("Alpha:", args.alpha)

    friedman_rows  = []
    rank_tables    = []
    pairwise_tables = []
    wins_tables    = []
    nemenyi_matrices: dict[str, pd.DataFrame] = {}

    for metric in metrics:
        print(f"\n=== Métrica: {metric} ===")

        wide = aggregate_by_dataset(df, metric=metric, methods=methods)
        wide.to_csv(args.output_dir / f"matrix_dataset_method_{sanitize_filename(metric)}.csv")

        friedman_rows.append(friedman_test(wide, metric))
        rank_tables.append(rank_methods(wide, metric))

        if args.wilcoxon:
            pairwise_tables.append(paired_wilcoxon_all_pairs(
                wide=wide, metric=metric, alpha=args.alpha,
                reference_method=args.reference_method,
            ))
        else:
            print(f"  [Wilcoxon] Desabilitado. Use --wilcoxon para ativar.")

        wins_tables.append(wins_by_dataset(wide, metric))

        nemenyi_matrix, status = try_nemenyi(
            wide=wide, metric=metric, output_dir=args.output_dir,
            methods=methods, alpha=args.alpha,
        )
        print(f"  {status}")
        if nemenyi_matrix is not None:
            nemenyi_matrices[metric] = nemenyi_matrix

    # Consolidação dos resultados
    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df["Significant"] = friedman_df["P_Value"] < args.alpha
    friedman_df = friedman_df.sort_values(["Metric"])

    rank_df    = pd.concat(rank_tables, ignore_index=True)
    wins_df    = pd.concat(wins_tables, ignore_index=True)

    wins_count_df = (
        wins_df.groupby(["Metric", "Best_Method"], as_index=False)
        .size()
        .rename(columns={"size": "Wins"})
        .sort_values(["Metric", "Wins", "Best_Method"], ascending=[True, False, True])
    )

    # Salva CSVs
    friedman_df.to_csv(args.output_dir / "statistical_friedman_summary.csv", index=False)
    rank_df.to_csv(args.output_dir / "statistical_average_ranks.csv", index=False)
    wins_df.to_csv(args.output_dir / "statistical_best_method_by_dataset.csv", index=False)
    wins_count_df.to_csv(args.output_dir / "statistical_win_counts.csv", index=False)

    pairwise_df = None
    if args.wilcoxon and pairwise_tables:
        pairwise_df = pd.concat(pairwise_tables, ignore_index=True)
        pairwise_df.to_csv(args.output_dir / "statistical_pairwise_wilcoxon_holm.csv", index=False)

    # LaTeX
    if args.latex:
        save_latex_tables(
            friedman_df=friedman_df,
            rank_df=rank_df,
            pairwise_df=pairwise_df,
            output_dir=args.output_dir,
        )

    # Gráficos
    print("\n=== Gerando gráficos ===")
    n_datasets_total = int(friedman_df["N_Datasets"].max()) if not friedman_df.empty else 30

    for metric in metrics:
        rank_subset = rank_df[rank_df["Metric"] == metric]
        n_ds = int(rank_subset["N_Datasets"].iloc[0]) if not rank_subset.empty else n_datasets_total
        nemenyi_matrix = nemenyi_matrices.get(metric)

        plot_cd_diagram(
            rank_df=rank_df,
            metric=metric,
            n_datasets=n_ds,
            alpha=args.alpha,
            nemenyi_matrix=nemenyi_matrix,
            output_dir=args.output_dir,
        )

        if nemenyi_matrix is not None:
            plot_nemenyi_heatmap(
                nemenyi_matrix=nemenyi_matrix,
                metric=metric,
                alpha=args.alpha,
                output_dir=args.output_dir,
            )

    plot_win_counts(
        wins_count_df=wins_count_df,
        metrics=metrics,
        output_dir=args.output_dir,
    )

    print("\nArquivos principais gerados:")
    print(f"  - {args.output_dir / 'statistical_friedman_summary.csv'}")
    print(f"  - {args.output_dir / 'statistical_average_ranks.csv'}")
    print(f"  - {args.output_dir / 'statistical_best_method_by_dataset.csv'}")
    print(f"  - {args.output_dir / 'statistical_win_counts.csv'}")
    if args.wilcoxon:
        print(f"  - {args.output_dir / 'statistical_pairwise_wilcoxon_holm.csv'}")
    else:
        print("  [Wilcoxon] Não executado. Use --wilcoxon para ativar.")
    print(f"  - cd_diagram_<metrica>.png  (por métrica)")
    print(f"  - nemenyi_heatmap_<metrica>.png  (se scikit-posthocs instalado)")
    print(f"  - win_counts_<metrica>.png  (por métrica)")


if __name__ == "__main__":
    main()