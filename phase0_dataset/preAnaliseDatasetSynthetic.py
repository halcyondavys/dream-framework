from main.configs import *
from phase1_global.loadData import *

import pandas as pd
import numpy  as np

from scipy.stats        import kurtosis, skew
from sklearn.neighbors  import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

# Import do gerador fatorial — necessário para F1 (gating) e F3b (bootstrap),
# que dependem de kappa/delta/n_samples originais, não apenas do .data salvo.
from data.Gerador_synthetic import make_cell, F1_LEVELS, F2_LEVELS, F3_LEVELS

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================
#   Configurações
# ============================

datasets_used = datasets_synthetic()
paths = path_DREAM()
results_path_phase1 = paths["results_path_phase1"]
output_csv_path     = paths["output_csv_path_phase1"]

os.makedirs(output_csv_path, exist_ok=True)

# ============================
#   Funções Estatísticas (originais)
# ============================

def calculate_cv(data, epsilon=1e-9):
    mean_abs = np.abs(np.mean(data))
    return np.std(data) / (mean_abs + epsilon)

def calculate_sorting(data):
    return np.std(data) / (np.max(data) - np.min(data))

def calculate_n3(X, y):
    epsilon = 1e-9
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    distances, _ = nn.kneighbors(X)
    var_y = np.var(y)
    if var_y == 0:
        var_y = epsilon
    return np.mean(distances[:, 1]) / var_y

def calculate_mean_feature_correlation(features):
    corr_matrix = np.corrcoef(features.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    upper_triangle = np.triu_indices_from(corr_matrix, k=1)
    return np.mean(np.abs(corr_matrix[upper_triangle]))

def calculate_skewness(data):
    return skew(data)

def calculate_kurtosis(data):
    return kurtosis(data, fisher=False)

def calculate_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    return np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))


# ============================
#   Funções Estatísticas (novas — validação F1 e F3b)
# ============================

def parse_cell_name(dataset_name):
    """
    Extrai (f1_name, f2_name, f3_name) a partir do nome da célula,
    ex.: 'C10-discreto-alto-baixo' -> ('discreto', 'alto', 'baixo').
    Assume o padrão fixo Cxx-F1-F2-F3 definido em Generate_synthetic_fatorial.py.
    """
    parts = dataset_name.split('-')
    return parts[1], parts[2], parts[3]


def calculate_gating_entropy(X_relevant, kappa):
    """
    Validação de F1 (discretização de regime).
    Entropia média dos pesos de gating w(x) sobre as instâncias reais
    do dataset (não uma amostra sintética nova — usa o X já carregado,
    restrito às 3 colunas relevantes, para refletir a distribuição real
    gerada, inclusive sob F3a onde há colunas irrelevantes concatenadas).

    Entropia baixa  -> pesos quase one-hot -> regime discreto confirmado.
    Entropia alta   -> pesos difusos       -> regime contínuo confirmado.
    Teto teórico: log(3) ≈ 1,0986 (3 especialistas, pesos uniformes).
    """
    x1, x2, x3 = X_relevant[:, 0], X_relevant[:, 1], X_relevant[:, 2]
    z = np.column_stack([
        kappa * (x1 - 0.5),
        -kappa * np.abs(x2 - 0.5),
        kappa * (x3 - 0.5),
    ])
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    ent = -np.sum(w * np.log(w + 1e-12), axis=1)
    return float(ent.mean()), float(ent.std())


def calculate_knn_bootstrap_variance(f1_name, f2_name, f3_name,
                                      k=10, n_bootstrap=30,
                                      query_point=None, base_seed=5000):
    """
    Validação de F3b (flutuação local sob amostra finita).
    Regenera a MESMA célula (mesmo kappa, delta, n_samples, n_irrelevant)
    com n_bootstrap sementes distintas, e mede a variância da estimativa
    de competência local (proxy: média de y entre os k vizinhos mais
    próximos de um ponto de consulta fixo) entre as realizações.

    Ponto de consulta fixo (centro do hipercubo unitário) garante
    comparabilidade entre células — nenhuma célula é favorecida por
    estar mais densamente amostrada perto do ponto escolhido.

    Variância alta -> estimativa de competência instável sob F3b,
    consistente com Stone (1977) para K fixo e N pequeno.
    """
    kappa = F1_LEVELS[f1_name]
    delta = F2_LEVELS[f2_name]
    f3_params = F3_LEVELS[f3_name]

    if query_point is None:
        query_point = np.array([0.5, 0.5, 0.5])

    estimates = []
    for b in range(n_bootstrap):
        X, y = make_cell(
            kappa=kappa, delta=delta,
            n_samples=f3_params["n_samples"],
            n_irrelevant=f3_params["n_irrelevant"],
            noise=0.05, random_state=base_seed + b,
        )
        X_rel = X[:, :3]  # apenas atributos relevantes, exclui F3a se presente
        k_eff = min(k, len(X_rel))
        nn = NearestNeighbors(n_neighbors=k_eff).fit(X_rel)
        _, idx = nn.kneighbors(query_point.reshape(1, -1))
        estimates.append(np.mean(y[idx[0]]))

    return float(np.var(estimates))


# =============================
#   Funções de contribuição (originais, inalteradas)
# =============================

def contribution_binary(value, homog_condition):
    return -1 if homog_condition(value) else 1

def contribution_ternary(value, homog_condition, heterog_condition):
    if homog_condition(value):
        return -1
    elif heterog_condition(value):
        return 1
    return 0


# =============================
#   Classificação do Dataset (original, sem thresholds novos)
# =============================
# NOTA: não adiciono classificação binária/ternária para gating_entropy_mean
# ou knn_bootstrap_variance ainda — os limiares de cv/sorting/n3/etc. foram
# calibrados empiricamente na literatura e nos dados reais (Cap. 5). Para
# os dois novos indicadores, ainda não há distribuição de referência.
# Recomendo calcular sobre as 12 células primeiro, observar a separação
# empírica entre níveis "continuo"/"discreto" e "baixo"/"f3b", e só então
# definir limiares — do contrário o corte seria arbitrário.

def classify_row(row):
    n_total = row['n_total']
    outlier_ratio = row['num_outliers'] / n_total if n_total > 0 else 0

    contributions = {
        "cv_score": contribution_binary(row['cv'], lambda x: x < 0.5),
        "sorting_score": contribution_ternary(row['sorting'], lambda x: x < 0.12, lambda x: x > 0.22),
        "kurtosis_score": contribution_binary(row['kurtosis'], lambda x: x < 3),
        "corr_score": contribution_ternary(row['mean_corr'], lambda x: x > 0.7, lambda x: x < 0.3),
        "n3_score_class": contribution_ternary(row['n3_score'], lambda x: x < 0.15, lambda x: x > 0.25),
        "outlier_score": contribution_ternary(outlier_ratio, lambda x: x < 0.05, lambda x: x > 0.10),
    }

    heterogeneity_score = sum(contributions.values())

    if heterogeneity_score < 0:
        perfil = 'Homogêneo'
    elif heterogeneity_score > 0:
        perfil = 'Heterogêneo'
    else:
        perfil = 'Híbrido'

    return pd.Series({
        **contributions,
        "outlier_ratio": outlier_ratio,
        "heterogeneity_score": heterogeneity_score,
        "Perfil_Dataset": perfil
    })


# =============================
#   Execução da Pré-Análise
# =============================

summary_results = []
profile_results = []

for dataset_name in datasets_used:
    print(f'🔍 Analisando dataset: {dataset_name}')

    data, labels = load_data(f'../data/synthetic/{dataset_name}.data')

    features = data
    target = labels

    n_total = len(target)
    n_features = data.shape[1]

    f1_name, f2_name, f3_name = parse_cell_name(dataset_name)
    X_relevant = features[:, :3]  # exclui atributos irrelevantes de F3a, se houver

    ent_mean, ent_std = calculate_gating_entropy(X_relevant, F1_LEVELS[f1_name])
    boot_var = calculate_knn_bootstrap_variance(f1_name, f2_name, f3_name)

    profile = {
        "Dataset": dataset_name,
        "F1": f1_name, "F2": f2_name, "F3": f3_name,
        "cv": calculate_cv(target),
        "sorting": calculate_sorting(target),
        "skewness": calculate_skewness(target),
        "kurtosis": calculate_kurtosis(target),
        "mean_corr": calculate_mean_feature_correlation(features),
        "n3_score": calculate_n3(features, target),
        "num_outliers": calculate_outliers(target),
        "n_total": n_total,
        "gating_entropy_mean": ent_mean,   # validação F1
        "gating_entropy_std": ent_std,
        "knn_bootstrap_variance": boot_var,  # validação F3b
    }

    summary_results.append(profile)

    profile_results.append({
        "Dataset": dataset_name,
        "n_instances": n_total,
        "n_features": n_features + 1
    })


# =============================
#    Consolidação Final
# =============================

summary_df = pd.DataFrame(summary_results)

classification_df = summary_df.apply(classify_row, axis=1)
summary_df = pd.concat([summary_df, classification_df], axis=1)

profile_df = pd.DataFrame(profile_results)

summary_df.to_csv(output_csv_path + "statistical_summary_classified.csv", index=False)
profile_df.to_csv(output_csv_path + "statistical_profile.csv", index=False)

print(f"\n✅ Arquivos salvos em: {output_csv_path}")
print(summary_df.to_string(index=False))