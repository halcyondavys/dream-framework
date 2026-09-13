import numpy as np


def knn(dataSet, point, K):
    """
    Versão Otimizada para o DREAM.
    Mantém a lógica de excluir a própria instância para evitar bias local.
    """
    # 1. Cálculo eficiente da distância Euclidiana
    # np.linalg.norm é otimizado internamente
    dist = np.linalg.norm(dataSet - point, axis=1)

    # 2. Filtro de distância zero com tolerância (Epsilon)
    # Isso garante que não pegaremos a própria instância no cálculo de competência
    is_not_self = dist > 1e-12

    # Se por algum motivo não sobrarem pontos, retornamos vazio (o caller trata)
    if not np.any(is_not_self):
        return np.array([]), np.array([])

    # Filtramos os dados válidos
    valid_indices = np.where(is_not_self)[0]
    valid_distances = dist[is_not_self]

    # 3. Seleção dos K vizinhos (mais rápido que argsort completo)
    k_actual = min(K, len(valid_distances))

    # argpartition coloca os k menores no início, sem ordenar o resto
    rel_idx = np.argpartition(valid_distances, k_actual - 1)[:k_actual]

    # Como o IDW (Inverse Distance Weighting) depende da ordem,
    # ordenamos apenas os K selecionados
    sorted_rel_idx = rel_idx[np.argsort(valid_distances[rel_idx])]

    # 4. Retorno dos índices originais e distâncias
    return valid_distances[sorted_rel_idx], valid_indices[sorted_rel_idx]