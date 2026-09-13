import numpy as np

def build_folds(all_labels, folds, Tr, Va, Te):
    if Tr + Te + Va != folds:
        raise ValueError("Soma Tr + Va + Te deve ser igual ao número de folds")

    N = len(all_labels)
    indices = np.arange(N)
    np.random.shuffle(indices)

    fold_size = N // folds
    split_indices = [indices[i * fold_size: (i + 1) * fold_size] for i in range(folds - 1)]
    split_indices.append(indices[(folds - 1) * fold_size:])  # último com o resto

    train_index = []
    valid_index = []
    test_index = []

    for i in range(folds):
        dados_folds = list(range(i, folds)) + list(range(0, i))
        train = np.zeros(N, dtype=bool)
        valid = np.zeros(N, dtype=bool)
        test = np.zeros(N, dtype=bool)

        for x in range(Tr):
            train[split_indices[dados_folds[x]]] = True
        for y in range(x + 1, x + Te + 1):
            test[split_indices[dados_folds[y]]] = True
        for z in range(y + 1, y + Va + 1):
            valid[split_indices[dados_folds[z]]] = True

        train_index.append(train)
        valid_index.append(valid)
        test_index.append(test)

    return np.array(train_index).T, np.array(valid_index).T, np.array(test_index).T
