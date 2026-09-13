# shared_base.py — módulo interno compartilhado pelos scripts de comparação
# NÃO executar diretamente. Importado pelos scripts 03–08.
# Contém: rcParams, constantes tipográficas, utilitários, funções estatísticas e de plot.

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# ---------------------------------------------------------------------------
# Tipografia — compatível com Times New Roman (tese ABNT/PPGEc-UPE)
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

FIGURE_WIDTH_IN  = 5.91   # largura útil A4/ABNT
FIGURE_HEIGHT_IN = 4.13

METHOD_DISPLAY_LABELS: dict[str, str] = {
    "DREAM_DS":    "DREAM-DS",
    "DREAM_DW":    "DREAM-DW",
    "DREAM_DWS":   "DREAM-DWS",
    "HET_DREAM_AVG": "HET-DREAM",
    "TOP3_AVG":    "TOP3-AVG",
    "TOP5_AVG":    "TOP5-AVG",
    "TOP_ALL_AVG": "TOP-ALL",
    "STACKING_ET":  "Stack-ET",
    "STACKING_RF":  "Stack-RF",
    "STACKING_XGB": "Stack-XGB",
}

DEFAULT_METRICS       = ["MSE", "RMSE", "MAE", "MedAE", "R2"]
DEFAULT_ORACLE_METHODS = {"ORACLE_BASE", "KMEN_ORACLE"}
HIGHER_IS_BETTER      = {"R2"}

# Cores fixas por método para consistência entre figuras
METHOD_COLORS: dict[str, str] = {
    "DREAM_DS":      "#2A6FAC",
    "DREAM_DW":      "#E07B00",
    "DREAM_DWS":     "#27AE60",
    "HET_DREAM_AVG": "#8E44AD",
    "TOP3_AVG":      "#C0392B",
    "TOP5_AVG":      "#16A085",
    "TOP_ALL_AVG":   "#D35400",
    "STACKING_ET":   "#2C3E50",
    "STACKING_RF":   "#7F8C8D",
    "STACKING_XGB":  "#F39C12",
}
FALLBACK_COLORS = ["#1ABC9C","#E74C3C","#3498DB","#9B59B6","#F1C40F"]


def method_label(method: str) -> str:
    return METHOD_DISPLAY_LABELS.get(method, method.replace("_", "-"))


def method_color(method: str, idx: int = 0) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def sanitize_filename(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Colunas ausentes no CSV: " + ", ".join(missing))


def metric_direction(metric: str) -> str:
    return "max" if metric.upper() in HIGHER_IS_BETTER else "min"


def load_and_merge(
    paths: list[Path],
    methods: list[str],
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carrega e concatena múltiplos CSVs, filtrando métodos e datasets.
    """
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")
        df = pd.read_csv(p)
        validate_columns(df, ["Dataset", "Exec", "Fold", "Method"])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Filtrar métodos desejados
    combined = combined[combined["Method"].isin(methods)]

    # Filtrar datasets se especificado
    if datasets:
        combined = combined[combined["Dataset"].isin(datasets)]

    return combined


def aggregate_by_dataset(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
) -> pd.DataFrame:
    """
    Retorna matriz wide: linhas=datasets, colunas=métodos, valores=média(metric).
    Descarta datasets sem todos os métodos presentes.
    """
    validate_columns(df, ["Dataset", "Exec", "Fold", "Method", metric])

    data = df.loc[df["Method"].isin(methods), ["Dataset", "Method", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])

    grouped = data.groupby(["Dataset", "Method"], as_index=False)[metric].mean()
    wide = grouped.pivot(index="Dataset", columns="Method", values=metric)
    wide = wide.dropna(axis=0, how="any")
    methods_present = [m for m in methods if m in wide.columns]
    wide = wide[methods_present]

    if wide.shape[0] < 2:
        raise ValueError(f"Métrica {metric}: menos de 2 datasets completos após filtros.")
    if wide.shape[1] < 2:
        raise ValueError(f"Métrica {metric}: menos de 2 métodos completos após filtros.")

    return wide


def rank_methods(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    ascending = metric_direction(metric) == "min"
    ranks = wide.rank(axis=1, method="average", ascending=ascending)
    return pd.DataFrame({
        "Metric": metric,
        "Method": ranks.columns,
        "Average_Rank": ranks.mean(axis=0).values,
        "Median_Rank":  ranks.median(axis=0).values,
        "Std_Rank":     ranks.std(axis=0, ddof=1).values,
        "N_Datasets":   wide.shape[0],
    }).sort_values(["Average_Rank", "Method"])


def friedman_test(wide: pd.DataFrame, metric: str) -> dict:
    arrays = [wide[col].to_numpy() for col in wide.columns]
    stat, p = friedmanchisquare(*arrays)
    return {
        "Metric": metric,
        "N_Datasets": int(wide.shape[0]),
        "N_Methods":  int(wide.shape[1]),
        "Friedman_Statistic": float(stat),
        "P_Value": float(p),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = (m - rank) * p_values[idx]
        running_max = max(running_max, adj_p)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def paired_wilcoxon_all_pairs(
    wide: pd.DataFrame,
    metric: str,
    alpha: float,
    reference_method: str | None = None,
) -> pd.DataFrame:
    direction = metric_direction(metric)
    rows = []
    for method_a, method_b in itertools.combinations(wide.columns, 2):
        a = wide[method_a].astype(float)
        b = wide[method_b].astype(float)
        diff = a - b
        if np.allclose(diff.to_numpy(), 0.0):
            stat, p_value = 0.0, 1.0
        else:
            try:
                stat, p_value = wilcoxon(a, b, alternative="two-sided",
                                         zero_method="wilcox", mode="auto")
            except ValueError:
                stat, p_value = np.nan, np.nan
        if direction == "min":
            wins_a = int((a < b).sum())
            wins_b = int((b < a).sum())
            better = method_a if a.mean() < b.mean() else method_b
        else:
            wins_a = int((a > b).sum())
            wins_b = int((b > a).sum())
            better = method_a if a.mean() > b.mean() else method_b
        rows.append({
            "Metric": metric,
            "Method_A": method_a, "Method_B": method_b,
            "Reference_Comparison": bool(reference_method and
                (method_a == reference_method or method_b == reference_method)),
            "N_Datasets": int(wide.shape[0]),
            "Wilcoxon_Statistic": float(stat) if not pd.isna(stat) else np.nan,
            "P_Value": float(p_value) if not pd.isna(p_value) else np.nan,
            "Mean_A": float(a.mean()), "Mean_B": float(b.mean()),
            "Median_A": float(a.median()), "Median_B": float(b.median()),
            "Wins_A": wins_a, "Wins_B": wins_b,
            "Ties": int(np.isclose(a, b).sum()),
            "Better_By_Mean": better, "Direction": direction,
        })
    result = pd.DataFrame(rows)
    valid_mask = result["P_Value"].notna()
    adjusted = [np.nan] * len(result)
    if valid_mask.any():
        adj_valid = holm_adjust(result.loc[valid_mask, "P_Value"].tolist())
        for idx, adj_p in zip(result.index[valid_mask].tolist(), adj_valid):
            adjusted[idx] = adj_p
    result["P_Adjusted_Holm"] = adjusted
    result["Significant_Holm"] = result["P_Adjusted_Holm"] < alpha
    return result.sort_values(["Metric", "P_Adjusted_Holm", "P_Value"])


def wins_by_dataset(wide: pd.DataFrame, metric: str) -> pd.DataFrame:
    direction = metric_direction(metric)
    best_method = wide.idxmin(axis=1) if direction == "min" else wide.idxmax(axis=1)
    best_value  = wide.min(axis=1)    if direction == "min" else wide.max(axis=1)
    return pd.DataFrame([{
        "Metric": metric, "Dataset": ds,
        "Best_Method": best_method[ds], "Best_Value": float(best_value[ds]),
        "Direction": direction,
    } for ds in wide.index])


def try_nemenyi(
    wide: pd.DataFrame, metric: str, output_dir: Path,
) -> tuple[pd.DataFrame | None, str]:
    try:
        import scikit_posthocs as sp
    except ImportError:
        return None, "scikit-posthocs não instalado. Execute: pip install scikit-posthocs"
    data = wide.copy()
    if metric_direction(metric) == "max":
        data = -data
    nemenyi = sp.posthoc_nemenyi_friedman(data.to_numpy())
    nemenyi.index   = wide.columns
    nemenyi.columns = wide.columns
    nemenyi.to_csv(output_dir / f"nemenyi_{sanitize_filename(metric)}.csv")
    return nemenyi, f"Nemenyi salvo: nemenyi_{sanitize_filename(metric)}.csv"


def _nemenyi_cd_value(
    n_methods: int,
    n_datasets: int,
    alpha: float = 0.05,
) -> float:
    """
    Calcula a diferença crítica do pós-teste de Nemenyi.

    A implementação utiliza os valores críticos apresentados por
    Demšar (2006) para alpha = 0.05 e comparações entre 2 e 10 métodos.

    Parameters
    ----------
    n_methods : int
        Número de métodos comparados.

    n_datasets : int
        Número de datasets utilizados como blocos experimentais.

    alpha : float, default=0.05
        Nível de significância. A tabela implementada está restrita
        a alpha = 0.05.

    Returns
    -------
    float
        Valor da diferença crítica do teste de Nemenyi.
    """

    if n_datasets < 2:
        raise ValueError(
            "O cálculo da diferença crítica exige pelo menos 2 datasets."
        )

    if not np.isclose(alpha, 0.05):
        raise ValueError(
            "A tabela de valores críticos implementada está disponível "
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
            "O cálculo da diferença crítica está configurado apenas "
            "para comparações entre 2 e 10 métodos."
        )

    q_alpha = q_alpha_table[n_methods]

    standard_error = np.sqrt(
        n_methods * (n_methods + 1)
        / (6 * n_datasets)
    )

    return q_alpha * standard_error


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_global_boxplot(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    output_path: Path,
    log_scale: bool = False,
) -> None:
    data = df.loc[df["Method"].isin(methods), ["Dataset", "Method", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    agg = data.groupby(["Dataset","Method"], as_index=False)[metric].mean()

    values         = [agg.loc[agg["Method"]==m, metric].dropna().to_numpy() for m in methods]
    display_labels = [method_label(m) for m in methods]
    colors         = [method_color(m, i) for i, m in enumerate(methods)]

    # Largura dinâmica para muitos métodos
    fig_w = max(FIGURE_WIDTH_IN, 0.85 * len(methods) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, FIGURE_HEIGHT_IN))

    from matplotlib.lines import Line2D
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

    ax.set_xlabel("Método")
    ax.set_ylabel(metric)
    tick_rot = 30 if len(methods) > 4 else 0
    ax.tick_params(axis="x", rotation=tick_rot)

    legend_elements = [
        Line2D([0],[0], color="#E07B00", linewidth=1.5, label="Mediana"),
        Line2D([0],[0], color="#2A7A2A", linewidth=1.2, linestyle="--", label="Média"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, framealpha=0.9)

    if log_scale and metric.upper() != "R2":
        pos = np.concatenate([v[v>0] for v in values if len(v[v>0])>0])
        if len(pos) > 0:
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cd_diagram(
    rank_df: pd.DataFrame, metric: str, n_datasets: int, alpha: float,
    nemenyi_matrix: pd.DataFrame | None, output_dir: Path,
) -> None:
    subset = rank_df[rank_df["Metric"] == metric].sort_values("Average_Rank")
    methods = subset["Method"].tolist()
    ranks   = subset["Average_Rank"].tolist()
    n_methods = len(methods)
    if n_methods < 2:
        return

    cd = _nemenyi_cd_value(n_methods, n_datasets, alpha)

    def _not_sig(m1: str, m2: str) -> bool:
        if nemenyi_matrix is not None and m1 in nemenyi_matrix.index and m2 in nemenyi_matrix.columns:
            return float(nemenyi_matrix.loc[m1, m2]) > alpha
        return abs(ranks[methods.index(m1)] - ranks[methods.index(m2)]) <= cd

    # Determina as cliques máximas de métodos sem diferença significativa.
    #
    # A ausência de significância não é transitiva. Por exemplo:
    # A pode não diferir de B e B pode não diferir de C,
    # embora A e C apresentem diferença significativa.
    #
    # Cada barra do diagrama deve conter somente métodos cujas
    # comparações par a par apresentem p > alpha.

    candidate_cliques: list[tuple[str, ...]] = []

    for group_size in range(2, len(methods) + 1):
        for candidate in itertools.combinations(methods, group_size):
            all_pairs_not_significant = all(
                _not_sig(method_a, method_b)
                for method_a, method_b in itertools.combinations(candidate, 2)
            )

            if all_pairs_not_significant:
                candidate_cliques.append(candidate)

    # Mantém apenas cliques máximas.
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

    # Remove duplicações preservando a ordem dos ranks.
    unique_cliques: list[list[str]] = []
    seen_cliques: set[tuple[str, ...]] = set()

    for clique in cliques:
        clique_key = tuple(clique)

        if clique_key not in seen_cliques:
            seen_cliques.add(clique_key)
            unique_cliques.append(clique)

    cliques = unique_cliques

    # Layout
    fig_h = max(FIGURE_HEIGHT_IN, 0.6 * n_methods + 1.8)
    fig_w = max(FIGURE_WIDTH_IN, 0.5 * n_methods + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(-0.5, n_methods + 1.0)
    ax.axis("off")

    rank_min, rank_max = min(ranks), max(ranks)
    rank_range = max(rank_max - rank_min, 1.0)

    def _rx(r): return 1.0 + (r - rank_min) / rank_range * (n_methods - 1)

    y_axis = n_methods * 0.5
    ax.axhline(y=y_axis, xmin=0.05, xmax=0.95, color="black", linewidth=1.2, zorder=1)

    # Barra CD
    x_cd_s, x_cd_e = _rx(rank_min), _rx(rank_min + cd)
    y_cd = n_methods * 0.94
    ax.annotate("", xy=(x_cd_e, y_cd), xytext=(x_cd_s, y_cd),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.text((x_cd_s + x_cd_e)/2, y_cd + 0.28, f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=9)

    # Distribuição alternada acima/abaixo
    top_m    = methods[: n_methods // 2 + n_methods % 2]
    bottom_m = methods[n_methods // 2 + n_methods % 2 :]
    method_y: dict[str, float] = {}
    for i, m in enumerate(top_m):    method_y[m] = y_axis + 0.6 + i * 0.55
    for i, m in enumerate(bottom_m): method_y[m] = y_axis - 0.6 - i * 0.55

    for m, r in zip(methods, ranks):
        x, y = _rx(r), method_y[m]
        is_top = m in top_m
        ax.plot([x, x], [y_axis, y], color="black", linewidth=0.8, zorder=2)
        ax.plot(x, y_axis, "o", color=method_color(m), markersize=5, zorder=3)
        ax.text(x, y + (0.05 if is_top else -0.05),
                f"{method_label(m)}\n(rank {r:.2f})",
                ha="center", va="bottom" if is_top else "top", fontsize=8.5)

    # Barras de não-significância
    clique_colors = ["#2A6FAC","#C0392B","#27AE60","#8E44AD","#E67E22","#16A085"]
    y_clique = y_axis - 0.22
    for ci, clique in enumerate(cliques):
        cr = [ranks[methods.index(m)] for m in clique]
        ax.plot([_rx(min(cr)) - 0.02, _rx(max(cr)) + 0.02], [y_clique - ci*0.12]*2,
                color=clique_colors[ci % len(clique_colors)],
                linewidth=4.0, solid_capstyle="round", zorder=4,
                label=f"Grupo {ci+1} (p > {alpha})")

    if cliques:
        ax.legend(loc="lower center", fontsize=8, frameon=True, framealpha=0.9,
                  ncol=min(len(cliques), 3))

    ax.set_title(f"CD Diagram — {metric} (α = {alpha})", pad=6)
    fig.tight_layout()
    out = output_dir / f"cd_diagram_{sanitize_filename(metric)}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] CD diagram: {out}")


def plot_nemenyi_heatmap(
    nemenyi_matrix: pd.DataFrame, metric: str, alpha: float, output_dir: Path,
) -> None:
    methods = list(nemenyi_matrix.columns)
    n = len(methods)
    labels = [method_label(m) for m in methods]
    matrix = nemenyi_matrix.to_numpy(dtype=float)

    fig_sz = max(FIGURE_WIDTH_IN, 0.9 * n + 0.8)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nemenyi", [(0.0,"#C0392B"),(alpha,"#F7DC6F"),(alpha+0.001,"#AED6F1"),(1.0,"#2874A6")])
    im = ax.imshow(matrix, cmap=cmap, norm=mpl.colors.Normalize(0,1), aspect="auto")

    for i in range(n):
        for j in range(n):
            val = matrix[i,j]
            text = "—" if i==j else f"{val:.3f}"
            color = "white" if (val < alpha*0.5 or val > 0.7) and i!=j else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p-valor (Nemenyi)", fontsize=9)
    cbar.ax.axhline(alpha, color="black", linewidth=1.2, linestyle="--")
    cbar.ax.text(1.5, alpha, f" α={alpha}", va="center", fontsize=8)
    ax.set_title(f"Nemenyi post-hoc — {metric}", pad=8)
    ax.spines[:].set_visible(False); ax.grid(False)
    fig.tight_layout()
    out = output_dir / f"nemenyi_heatmap_{sanitize_filename(metric)}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  [OK] Heatmap Nemenyi: {out}")


def plot_win_counts(
    wins_count_df: pd.DataFrame, metrics: list[str], methods: list[str], output_dir: Path,
) -> None:
    for metric in metrics:
        subset = wins_count_df[wins_count_df["Metric"] == metric].copy()
        if subset.empty:
            continue
        # Preservar ordem canônica dos métodos
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

        ax.set_xlabel("Método"); ax.set_ylabel("Nº de vitórias")
        ax.set_ylim(0, total + 2)
        tick_rot = 30 if len(display_labels) > 4 else 0
        ax.tick_params(axis="x", rotation=tick_rot)
        fig.tight_layout()
        out = output_dir / f"win_counts_{sanitize_filename(metric)}.png"
        fig.savefig(out); plt.close(fig)
        print(f"  [OK] Win counts: {out}")


def save_descriptive_summary(
    df: pd.DataFrame, metrics: list[str], methods: list[str], output_dir: Path,
) -> None:
    rows = []
    for metric in metrics:
        data = df.loc[df["Method"].isin(methods), ["Dataset","Method",metric]].copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        agg = data.groupby(["Dataset","Method"], as_index=False)[metric].mean()
        for method in methods:
            vals = agg.loc[agg["Method"]==method, metric].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                "Metric": metric, "Method": method, "N": int(len(vals)),
                "Mean": float(vals.mean()), "Median": float(vals.median()),
                "Std":  float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
                "Min":  float(vals.min()), "Q1": float(vals.quantile(0.25)),
                "Q3":   float(vals.quantile(0.75)), "Max": float(vals.max()),
            })
    pd.DataFrame(rows).to_csv(output_dir / "descriptive_summary.csv", index=False)


def save_latex_tables(
    friedman_df: pd.DataFrame, rank_df: pd.DataFrame,
    pairwise_df: pd.DataFrame | None, output_dir: Path,
    label_prefix: str = "comp",
) -> None:
    def _booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
        cols = list(df.columns)
        col_spec = "l" + "r" * (len(cols) - 1)

        def _fmt(v):
            if isinstance(v, float):
                return "--" if pd.isna(v) else f"{v:.4f}"
            if isinstance(v, bool):
                return "Sim" if v else "Não"
            return str(v)

        header = " & ".join(cols) + r" \\"
        body = "\n    ".join(" & ".join(_fmt(row[c]) for c in cols) + r" \\"
                             for _, row in df.iterrows())
        return (
            "\\begin{table}[htbp]\n  \\centering\n"
            f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            f"    \\toprule\n    {header}\n    \\midrule\n    {body}\n"
            "    \\bottomrule\n  \\end{tabular}\n"
            "  \\fonte{Elaborado pelo autor.}\n\\end{table}\n"
        )

    (output_dir / "friedman_summary.tex").write_text(_booktabs(
        friedman_df,
        caption=f"Resultados do teste de Friedman por métrica — {label_prefix}.",
        label=f"tab:friedman_{label_prefix}",
    ), encoding="utf-8")

    (output_dir / "average_ranks.tex").write_text(_booktabs(
        rank_df,
        caption=f"Ranking médio dos métodos por métrica — {label_prefix}.",
        label=f"tab:ranks_{label_prefix}",
    ), encoding="utf-8")

    if pairwise_df is not None:
        cols = ["Metric","Method_A","Method_B","P_Value","P_Adjusted_Holm",
                "Significant_Holm","Wins_A","Wins_B","Ties","Better_By_Mean"]
        available = [c for c in cols if c in pairwise_df.columns]
        (output_dir / "pairwise_wilcoxon_holm.tex").write_text(_booktabs(
            pairwise_df[available],
            caption=f"Wilcoxon pareado com correção de Holm — {label_prefix}.",
            label=f"tab:wilcoxon_{label_prefix}",
        ), encoding="utf-8")


def run_statistical_pipeline(
    df: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    output_dir: Path,
    alpha: float,
    wilcoxon_enabled: bool,
    reference_method: str | None,
    latex_enabled: bool,
    label_prefix: str,
    dpi: int,
) -> None:
    """
    Pipeline completo: Friedman → Nemenyi → gráficos → CSVs → LaTeX.
    Compartilhado por todos os scripts de comparação.
    """
    mpl.rcParams["savefig.dpi"] = dpi
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filtrar métricas disponíveis
    metrics = [m for m in metrics if m in df.columns]

    friedman_rows, rank_tables, pairwise_tables, wins_tables = [], [], [], []
    nemenyi_matrices: dict[str, pd.DataFrame] = {}

    for metric in metrics:
        print(f"\n  === {metric} ===")
        wide = aggregate_by_dataset(df, metric=metric, methods=methods)
        wide.to_csv(output_dir / f"matrix_{sanitize_filename(metric)}.csv")

        friedman_rows.append(friedman_test(wide, metric))
        rank_tables.append(rank_methods(wide, metric))

        if wilcoxon_enabled:
            pairwise_tables.append(paired_wilcoxon_all_pairs(
                wide=wide, metric=metric, alpha=alpha,
                reference_method=reference_method))
        else:
            print(f"    [Wilcoxon] desabilitado. Use --wilcoxon para ativar.")

        wins_tables.append(wins_by_dataset(wide, metric))

        nem, status = try_nemenyi(wide, metric, output_dir)
        print(f"    {status}")
        if nem is not None:
            nemenyi_matrices[metric] = nem

    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df["Significant"] = friedman_df["P_Value"] < alpha
    friedman_df = friedman_df.sort_values("Metric")

    rank_df    = pd.concat(rank_tables, ignore_index=True)
    wins_df    = pd.concat(wins_tables, ignore_index=True)
    wins_count_df = (wins_df.groupby(["Metric","Best_Method"], as_index=False)
                     .size().rename(columns={"size":"Wins"})
                     .sort_values(["Metric","Wins"], ascending=[True,False]))

    friedman_df.to_csv(output_dir / "friedman_summary.csv", index=False)
    rank_df.to_csv(output_dir / "average_ranks.csv", index=False)
    wins_df.to_csv(output_dir / "best_method_by_dataset.csv", index=False)
    wins_count_df.to_csv(output_dir / "win_counts.csv", index=False)

    pairwise_df = None
    if wilcoxon_enabled and pairwise_tables:
        pairwise_df = pd.concat(pairwise_tables, ignore_index=True)
        pairwise_df.to_csv(output_dir / "pairwise_wilcoxon_holm.csv", index=False)

    save_descriptive_summary(df, metrics=metrics, methods=methods, output_dir=output_dir)

    if latex_enabled:
        save_latex_tables(friedman_df, rank_df, pairwise_df, output_dir, label_prefix)

    # Gráficos estatísticos
    print("\n  Gerando gráficos...")
    n_ds_max = int(friedman_df["N_Datasets"].max()) if not friedman_df.empty else 30
    for metric in metrics:
        rs = rank_df[rank_df["Metric"] == metric]
        n_ds = int(rs["N_Datasets"].iloc[0]) if not rs.empty else n_ds_max
        nem = nemenyi_matrices.get(metric)
        plot_cd_diagram(rank_df, metric, n_ds, alpha, nem, output_dir)
        if nem is not None:
            plot_nemenyi_heatmap(nem, metric, alpha, output_dir)

    plot_win_counts(wins_count_df, metrics, methods, output_dir)

    print(f"\n  Resultados em: {output_dir}")
