from main.configs import *
from phase1_global.loadData import *

import pandas as pd
import numpy  as np

from scipy.stats        import kurtosis, skew
from sklearn.neighbors  import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================
#   Configurações
# ============================

#datasets_used = datasets_used_reduzido()
datasets_used = datasets_used()

paths = path_DREAM()

results_path_phase1 = paths["results_path_phase1"]
output_csv_path     = paths["output_csv_path_phase1"]

os.makedirs(output_csv_path, exist_ok=True)

# ============================
#   Funções Estatísticas
# ============================

def calculate_cv(data, epsilon=1e-9):
    """
    Coeficiente de Variação ajustado para evitar valores negativos.
    Calculado como desvio padrão dividido pelo valor absoluto da média.
    """
    mean_abs = np.abs(np.mean(data))
    return np.std(data) / (mean_abs + epsilon)

def calculate_sorting(data):
    """
    Sorting baseado no desvio padrão.
    Mede a dispersão absoluta dos valores em torno da média.
    """
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
    return kurtosis(data, fisher=False) #curtose passa a ser a curtose de Pearson, na qual a distribuição normal tem valor 3.


def calculate_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    return np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))


# =============================
#   Funções de contribuição
# =============================

def contribution_binary(value, homog_condition):
    """
    Métrica sem zona de incerteza.
    Retorna:
    -1 -> homogêneo
    +1 -> heterogêneo
    """
    return -1 if homog_condition(value) else 1


def contribution_ternary(value, homog_condition, heterog_condition):
    """
    Métrica com zona de incerteza.
    Retorna:
    -1 -> homogêneo
     0 -> incerto / híbrido local
    +1 -> heterogêneo
    """
    if homog_condition(value):
        return -1
    elif heterog_condition(value):
        return 1
    return 0


# =============================
#   Classificação do Dataset
# =============================

def classify_row(row):
    n_total = row['n_total']
    outlier_ratio = row['num_outliers'] / n_total if n_total > 0 else 0

    contributions = {
        # CV: sem zona de incerteza
        "cv_score": contribution_binary(
            row['cv'],
            lambda x: x < 0.5
        ),

        # Sorting: com zona de incerteza
        "sorting_score": contribution_ternary(
            row['sorting'],
            lambda x: x < 0.12,
            lambda x: x > 0.22
        ),

        # Kurtosis: sem zona de incerteza
        "kurtosis_score": contribution_binary(
            row['kurtosis'],
            lambda x: x < 3
        ),

        # Correlação média: com zona de incerteza
        "corr_score": contribution_ternary(
            row['mean_corr'],
            lambda x: x > 0.7,
            lambda x: x < 0.3
        ),

        # N3: com zona de incerteza
        "n3_score_class": contribution_ternary(
            row['n3_score'],
            lambda x: x < 0.15,
            lambda x: x > 0.25
        ),

        # Outliers: com zona de incerteza
        "outlier_score": contribution_ternary(
            outlier_ratio,
            lambda x: x < 0.05,
            lambda x: x > 0.10
        )
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

    data, labels = load_data(f'../data/real/{dataset_name}.data')

    features = data
    target = labels

    n_total = len(target)
    n_features = data.shape[1]

    profile = {
        "Dataset": dataset_name,
        "cv": calculate_cv(target),
        "sorting": calculate_sorting(target),
        "skewness": calculate_skewness(target),   # pode manter para auditoria futura
        "kurtosis": calculate_kurtosis(target),
        "mean_corr": calculate_mean_feature_correlation(features),
        "n3_score": calculate_n3(features, target),
        "num_outliers": calculate_outliers(target),
        "n_total": n_total
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