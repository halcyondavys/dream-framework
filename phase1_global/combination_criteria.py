import numpy as np
from sklearn.metrics       import mean_squared_error
from itertools             import product
import optuna

# ── Silenciar logs verbosos do Optuna em produção ───────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 🔹 Função para Determinar Threshold Dinâmico
def dynamic_threshold(scores, min_models=3, max_models=10, variation_limit=0.05):
    """
        Determina dinamicamente quantos modelos selecionar com base na variação
        relativa dos scores SDF ordenados.

        Parâmetros
        ----------
        scores : array-like de floats
            Scores SDF de cada modelo base (não precisa estar ordenado).
        min_models : int
            Número mínimo de modelos a selecionar.
        max_models : int
            Número máximo de modelos a selecionar.
        variation_limit : float
            Limiar de variação relativa entre scores consecutivos.
            Scores com variação <= variation_limit são agrupados.

        Retorna
        -------
        int : número de modelos a selecionar.
        """

    if len(scores) < min_models:
        return len(scores)

    sorted_scores = np.sort(scores)[::-1]
    variation     = np.abs(np.diff(sorted_scores) / (sorted_scores[:-1] + 1e-12))
    count_valid   = np.sum(variation <= variation_limit)
    num_models    = max(count_valid + 1, min_models)

    return int(min(num_models, max_models))

# ════════════════════════════════════════════════════════════════════════════
# SCORE COMBINADO (SDF)
# ════════════════════════════════════════════════════════════════════════════
def combined_score(
    div_corr,         # maior=melhor (já em [0,1] após nossa nova função)
    var_error,        # variância dos erros por modelo (maior=pior)
    mse_error,        # MSE por modelo (maior=pior)
    consensus_var,    # variância do desvio em relação ao consenso (maior=melhor)
    df_score,         # double fault (maior=melhor)
    alpha, beta, gamma, delta, epsilon,
    return_components: bool = False
):
    """
    Score combinado para ranquear modelos base do ensemble.

    O score integra métricas de:
    - diversidade estrutural
    - estabilidade do erro
    - acurácia
    - dispersão em relação ao consenso
    - double fault
    """
    # 1. Acurácia e estabilidade já ajustadas
    stability = 1.0 - var_error
    accuracy = 1.0 - mse_error

    # 2. Score final
    score = (
            alpha * div_corr +
            beta * stability +
            gamma * accuracy +
            delta * consensus_var +
            epsilon * df_score
    )

    if return_components:
        return score, {
            "div_corr": div_corr,
            "stability": stability,
            "accuracy": accuracy,
            "consensus_var": consensus_var,
            "df_score": df_score
        }

    return score

# ════════════════════════════════════════════════════════════════════════════
# FUNÇÃO DE CUSTO (compartilhada entre busca em grade e Optuna)
# ════════════════════════════════════════════════════════════════════════════
def _calculate_ensemble_mse(
    weights,
    metrics_data,
    K_select=5,
    selection_mode="fixed_k",
    min_models=3,
    max_models=10,
    variation_limit=0.05
):
    """
    Calcula o MSE do ensemble selecionado a partir dos pesos do SDF.

    Parâmetros:
    ----------
    weights: tuple/list
        Pesos do SDF (alpha, beta, gamma, delta, epsilon)
    metrics_data: dic
        Dicionário contendo métricas normalizadas e previsões de validação.
    K_select : int
        Número fixo de modelos a selecionar quando selection_mode="fixed_k".
    selection_mode : str
        Estratégia de seleção durante a otimização:
        - "fixed_k": seleciona os K melhores modelos.
        - "dynamic_threshold": usa a mesma regra dinâmica da seleção final.

    min_models, max_models, variation_limit :
        Parâmetros usados pelo dynamic_threshold.

    Retorna:
    -------
        float: Mean Squared Error do ensemble selecionado.
    """
    alpha, beta, gamma, delta, epsilon = weights

    # 1. Obter métricas pré-calculadas
    div_corr = metrics_data['div_corr']
    var_error = metrics_data['var_error']
    mse_error = metrics_data['mse_error']
    consensus_var = metrics_data['consensus_var']
    df_score = metrics_data['df_score']

    # 2. Calcular o Score de Diversidade Final (SDF)
    # A função combined_score já faz a normalização e a inversão
    sdf_scores = combined_score(
        div_corr=div_corr,
        var_error=var_error,
        mse_error=mse_error,
        consensus_var=consensus_var,
        df_score=df_score,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        epsilon=epsilon
    )

    # 3. Seleção dos K melhores modelos
    if selection_mode == "dynamic_threshold":
        num_models = dynamic_threshold(
            sdf_scores,
            min_models=min_models,
            max_models=max_models,
            variation_limit=variation_limit
        )
        selected_indices = np.argsort(sdf_scores)[-num_models:]

    elif selection_mode == "fixed_k":
        selected_indices = np.argsort(sdf_scores)[::-1][:K_select]

    else:
        raise ValueError(
            "selection_mode inválido. Use 'fixed_k' ou 'dynamic_threshold'."
        )


    # Adicionando o melhor modelo individual (o modelo com menor MSE original)
    index_better_model = np.argmin(metrics_data['mse_error'])
    if index_better_model not in selected_indices:
        selected_indices = np.append(selected_indices, index_better_model)

    selected_indices = np.unique(selected_indices)  # Garante unicidade

    # 4. Agregação do Ensemble (média simples)
    # Aqui usamos as previsões brutas do conjunto de validação, acumuladas em todas as execuções
    y_true = metrics_data['y_true_validation']
    preds_validation = metrics_data['preds_validation']

    # A matriz preds_validation deve ser a matriz de (amostras, modelos)
    ensemble_predictions = preds_validation[:, selected_indices].mean(axis=1)

    # 5. Avaliação do Ensemble (Função de Custo)
    ensemble_mse = mean_squared_error(y_true, ensemble_predictions)

    return ensemble_mse

# ════════════════════════════════════════════════════════════════════════════
# OTIMIZAÇÃO COM OPTUNA
# ════════════════════════════════════════════════════════════════════════════
def optimize_sdf_weights(
    metrics_data,
    K_select=5,
    method="optuna",
    step=0.1,
    n_trials=200,
    population_size=50,
    generations=100,
    random_state=42,
    selection_mode="dynamic_threshold"
):
    """
    Otimiza os pesos do SDF por diferentes estratégias.

    Métodos disponíveis:
    - "optuna": otimização bayesiana/TPE via Optuna.
    - "ga": algoritmo genético simples.
    - "grid-search": busca em grade original.
    """

    if method == "optuna":
        return optimize_sdf_weights_optuna(
            metrics_data=metrics_data,
            K_select=K_select,
            n_trials=n_trials,
            random_state=random_state,
            selection_mode=selection_mode
        )

    elif method == "ga":
        return optimize_sdf_weights_ga(
            metrics_data=metrics_data,
            K_select=K_select,
            population_size=population_size,
            generations=generations,
            random_state=random_state,
            selection_mode=selection_mode
        )

    elif method == "grid-search":
        return optimize_sdf_weights_grid_search(
            metrics_data=metrics_data,
            K_select=K_select,
            step=step
        )

    else:
        raise ValueError(
            "Método inválido. Use: 'optuna', 'ga' ou 'grid-search'."
        )

def optimize_sdf_weights_grid_search(metrics_data, K_select=5, step=0.1):
    """
    Executa uma busca em grade para encontrar os pesos do SDF que minimizam
    o MSE do ensemble selecionado.

    Parâmetros:
    ----------
    metrics_data : dict contendo todas as métricas médias (mean_final) e
                   os dados de validação consolidados (y_true e preds).
    K_select : int, número fixo de modelos a selecionar (excluindo o melhor modelo garantido).
    step : float, resolução da grade de busca (ex: 0.1, 0.05).

    Retorna:
    -------
    tuple: (best_weights, best_mse)
    """

    # Gerar pesos discretos (ex: [0.0, 0.1, 0.2, ..., 1.0])
    possible_weights = np.round(np.arange(0.0, 1.0 + step, step), 2)

    best_mse = np.inf
    best_weights = None

    print(f"\nIniciando otimização com K={K_select} e resolução de peso: {step}")

    # Otimização por Força Bruta (5 pesos)
    for weights_raw in product(possible_weights, repeat=5):
        # Condição de restrição: a soma dos pesos deve ser 1.0
        if not np.isclose(sum(weights_raw), 1.0):
            continue

        current_mse = _calculate_ensemble_mse(weights_raw, metrics_data, K_select)

        if current_mse < best_mse:
            best_mse = current_mse
            best_weights = weights_raw

    print("-" * 50)
    print(f"MSE Mínimo Encontrado: {best_mse:.6f}")
    print(f"Pesos Otimizados (α, β, γ, δ, ε): {best_weights}")
    print("-" * 50)

    return best_weights, best_mse

def optimize_sdf_weights_optuna(
    metrics_data,
    K_select=5,
    n_trials=200,
    random_state=42,
    selection_mode="dynamic_threshold",
    min_models=3,
    max_models=10,
    variation_limit=0.05
):
    """
    Otimiza os pesos do SDF utilizando Optuna.

    A restrição de soma dos pesos igual a 1 é garantida por normalização:
    os cinco valores são amostrados como variáveis contínuas positivas e,
    em seguida, normalizados para formar um vetor de pesos no simplex.

    Parâmetros
    ----------
    metrics_data : dict
        Métricas normalizadas e previsões de validação.

    K_select : int
        Número fixo de modelos, usado apenas se selection_mode="fixed_k".

    n_trials : int
        Número de tentativas da otimização.

    random_state : int
        Semente para reprodutibilidade.

    selection_mode : str
        - "fixed_k": otimiza considerando K fixo.
        - "dynamic_threshold": otimiza usando a regra dinâmica final do DREAM.

    Retorna
    -------
    tuple
        (best_weights, best_mse)
    """

    if optuna is None:
        raise ImportError(
            "Optuna não está instalado. Instale com: pip install optuna"
        )

    def objective(trial):
        raw_weights = np.array([
            trial.suggest_float("alpha_raw",   1e-8, 1.0),
            trial.suggest_float("beta_raw",    1e-8, 1.0),
            trial.suggest_float("gamma_raw",   1e-8, 1.0),
            trial.suggest_float("delta_raw",   1e-8, 1.0),
            trial.suggest_float("epsilon_raw", 1e-8, 1.0)
        ])

        weights = raw_weights / raw_weights.sum()

        mse = _calculate_ensemble_mse(
            weights=weights,
            metrics_data=metrics_data,
            K_select=K_select,
            selection_mode=selection_mode,
            min_models=min_models,
            max_models=max_models,
            variation_limit=variation_limit
        )

        return mse

    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    print(
        f"\nIniciando otimização Optuna | "
        f"trials={n_trials} | "
        f"selection_mode={selection_mode}"
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_raw = np.array([
        study.best_params["alpha_raw"],
        study.best_params["beta_raw"],
        study.best_params["gamma_raw"],
        study.best_params["delta_raw"],
        study.best_params["epsilon_raw"]
    ])

    best_weights = best_raw / best_raw.sum()
    best_weights = tuple(best_weights.tolist())
    best_mse = float(study.best_value)

    print("-" * 50)
    print(f"MSE mínimo encontrado: {best_mse:.6f}")
    print(f"Pesos otimizados Optuna (α, β, γ, δ, ε): {best_weights}")
    print("-" * 50)

    return best_weights, best_mse

def optimize_sdf_weights_ga(
    metrics_data,
    K_select=5,
    population_size=50,
    generations=100,
    mutation_rate=0.20,
    mutation_scale=0.10,
    elite_size=5,
    random_state=42,
    selection_mode="dynamic_threshold",
    min_models=3,
    max_models=10,
    variation_limit=0.05
):
    """
    Otimiza os pesos do SDF usando um Algoritmo Genético simples.

    Cada indivíduo representa um vetor de cinco pesos:
    alpha, beta, gamma, delta, epsilon.

    A restrição de soma igual a 1 é garantida por normalização após
    inicialização, cruzamento e mutação.

    Retorna
    -------
    tuple
        (best_weights, best_mse)
    """

    rng = np.random.default_rng(random_state)

    def normalize_weights(w):
        w = np.clip(w, 1e-8, None)
        return w / w.sum()

    def create_individual():
        return rng.dirichlet(np.ones(5))

    def fitness(individual):
        return _calculate_ensemble_mse(
            weights=individual,
            metrics_data=metrics_data,
            K_select=K_select,
            selection_mode=selection_mode,
            min_models=min_models,
            max_models=max_models,
            variation_limit=variation_limit
        )

    def tournament_selection(population, scores, tournament_size=3):
        candidates = rng.choice(len(population), size=tournament_size, replace=False)
        best_candidate = candidates[np.argmin(scores[candidates])]
        return population[best_candidate]

    def crossover(parent1, parent2):
        lambda_value = rng.random()
        child = lambda_value * parent1 + (1.0 - lambda_value) * parent2
        return normalize_weights(child)

    def mutate(individual):
        if rng.random() < mutation_rate:
            noise = rng.normal(0.0, mutation_scale, size=individual.shape)
            individual = individual + noise
        return normalize_weights(individual)

    population = np.array([create_individual() for _ in range(population_size)])

    best_weights = None
    best_mse = np.inf

    print(
        f"\nIniciando otimização GA | "
        f"population={population_size} | "
        f"generations={generations} | "
        f"selection_mode={selection_mode}"
    )

    for generation in range(generations):
        scores = np.array([fitness(individual) for individual in population])

        generation_best_idx = int(np.argmin(scores))
        generation_best_mse = float(scores[generation_best_idx])

        if generation_best_mse < best_mse:
            best_mse = generation_best_mse
            best_weights = population[generation_best_idx].copy()

        elite_indices = np.argsort(scores)[:elite_size]
        new_population = [population[idx].copy() for idx in elite_indices]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = np.array(new_population)

    best_weights = tuple(best_weights.tolist())

    print("-" * 50)
    print(f"MSE mínimo encontrado: {best_mse:.6f}")
    print(f"Pesos otimizados GA (α, β, γ, δ, ε): {best_weights}")
    print("-" * 50)

    return best_weights, best_mse