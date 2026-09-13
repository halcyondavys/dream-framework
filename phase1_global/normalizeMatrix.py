from main.configs import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def normalize_by_fold(X_train_raw, y_train_raw, X_valid_raw, y_valid_raw, X_test_raw, y_test_raw, method='minmax'):
    """
    Normaliza os dados de treino, validação e teste com base apenas nas estatísticas do treino (sem leakage).

    Parâmetros:
    - X_train_raw, y_train_raw: dados de treino
    - X_valid_raw, y_valid_raw: dados de validação
    - X_test_raw, y_test_raw: dados de teste
    - method: 'minmax', 'standard' ou 'robust'

    Retorna:
    - data_train, data_valid, data_test: arrays no formato (label | features)
    """

    # Seleciona o tipo de normalização
    if method == 'standard':
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
    elif method == 'robust':
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()
    else:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

    # Fit apenas nos dados de treino
    X_train = scaler_x.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

    # Transform nos dados de validação e teste
    X_valid = scaler_x.transform(X_valid_raw)
    y_valid = scaler_y.transform(y_valid_raw.reshape(-1, 1)).flatten()
    X_test  = scaler_x.transform(X_test_raw)
    y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

    # Reunir os conjuntos no formato (label | features)
    data_train = np.column_stack([y_train, X_train])
    data_valid = np.column_stack([y_valid, X_valid])
    data_test  = np.column_stack([y_test,  X_test])

    return data_train, data_valid, data_test

def normalize_matrix_outliers(old_data, old_label, ratio_outliers):
    # 1. Escolha inteligente do Scaler (Sua Heurística)
    if ratio_outliers > 0.05:
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()
    else:
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

    # 3. Ajuste e Transformação Global
    data_normalized   = scaler_x.fit_transform(old_data)
    labels_normalized = scaler_y.fit_transform(old_label.reshape(-1, 1)).flatten()

    return data_normalized, labels_normalized, scaler_x, scaler_y

def normalize_matrix(old_data, old_label):
    rd, cd = old_data.shape
    norm_data = np.zeros((rd, cd))

    rl = old_label.shape[0]
    norm_label = np.zeros((rl,))

    for i in range(cd):
        max_data_error = np.max(old_data[:, i])
        min_data_error = np.min(old_data[:, i])

        if max_data_error == min_data_error:
            norm_data[:, i] = old_data[:, i]
        else:
            norm_data[:, i] = (old_data[:, i] - min_data_error) / (max_data_error - min_data_error)

    max_label_error = np.max(old_label)
    min_label_error = np.min(old_label)

    if max_label_error == min_label_error:
        norm_label[:] = old_label[:]
    else:
        norm_label[:] = (old_label[:] - min_label_error) / (max_label_error - min_label_error)

    return norm_data, norm_label

