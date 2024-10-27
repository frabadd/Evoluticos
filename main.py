import signal
import sys
import pandas as pd
import json
import numpy as np
import time
# Variable global para controlar la interrupción
interrupted = False

# Función para manejar SIGINT (Ctrl + C)
def signal_handler(sig, frame):
    global interrupted
    interrupted = True

# Registramos la señal SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)

def load_data(train_file, test_file):
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

import numpy as np

def generate_population(num_individuos, num_atributos):
    mascaras = np.random.randint(2, size=(num_individuos, num_atributos))
    pesos = np.random.uniform(-1, 1, (num_individuos, num_atributos))
    
    suma_pesos = pesos.sum(axis=1, keepdims=True)   
    suma_pesos[suma_pesos == 0] = 1e-10
    pesos_normalizados = pesos / suma_pesos
    poblacion = np.stack((mascaras, pesos_normalizados), axis=1)
    return poblacion


def fitness_poblacion(poblacion, Xtrain, Ytrain, umbral=0.5, metrica='f1'):
    poblacion = np.asarray(poblacion)
    population = np.copy(poblacion)
    population[:, 0, :] = (population[:, 0, :] > 0.5).astype(int)
    population[:, 1, :] = population[:, 1, :] * population[:, 0, :]
    sums = np.sum(population[:, 1, :], axis=1, keepdims=True)
    population[:, 1, :] = np.divide(population[:, 1, :], sums, where=sums != 0)
    probs = np.dot(Xtrain, population[:, 1, :].T)
    y_pred = (probs >= umbral).astype(int)
    y_true = Ytrain
    scores = np.empty(y_pred.shape[1])

    if metrica == 'f1':
        for i in range(y_pred.shape[1]):
            tp = np.sum((y_true == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true == 1) & (y_pred[:, i] == 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall == 0:
                scores[i] = 0
            else:
                scores[i] = 2 * (precision * recall) / (precision + recall)
    elif metrica == 'acc':
        for i in range(y_pred.shape[1]):
            correct_predictions = np.sum(y_true == y_pred[:, i])
            scores[i] = correct_predictions / len(y_true)

    return scores

def fitness(individuo, Xtrain, Ytrain, umbral=0.5, metrica='f1'):
    individuo = np.copy(np.asarray(individuo))
    individuo[1] = individuo[1] * individuo[0] / (np.sum(individuo[1] * individuo[0]) + 1e-10)   
    y_true = Ytrain
    probs = np.dot(Xtrain.astype(np.float64), individuo[1, :].T.astype(np.float64))
    y_pred = (probs >= umbral).astype(int)

    if metrica == 'f1':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    elif metrica == 'acc':
        return np.sum(y_true == y_pred) / len(y_true)

def seleccion_torneo(fitnessPoblacion, poblacion, k):
    seleccionados = []
    for _ in range(len(poblacion)):
        indices_torneo = np.random.choice(len(poblacion), k, replace=False)
        ganador = indices_torneo[np.argmax(fitnessPoblacion[indices_torneo])]
        seleccionados.append(poblacion[ganador])
    return np.array(seleccionados)

def mutacion_gaussiana_normalizada(individuo, prob_mutacion=0.1, sigma=0.01):
    mutante = np.copy(individuo)
    for i in range(len(mutante[0])):
        if np.random.rand() < prob_mutacion:
            mutante[0, i] = 1 - mutante[0, i]
    for i in range(len(mutante[1])):
        if np.random.rand() < prob_mutacion:
            mutante[1, i] += np.random.normal(0, sigma)
    mutante[1] /= np.sum(np.abs(mutante[1]))
    return mutante

def crossover(padre1, padre2, alpha=0.5):
    hijo1, hijo2 = np.empty(padre1.shape), np.empty(padre1.shape)
    mask = np.random.rand(len(padre1[0])) > 0.5
    hijo1[0] = np.where(mask, padre1[0], padre2[0])
    hijo2[0] = np.where(~mask, padre1[0], padre2[0])
    alphas1 = np.random.uniform(-alpha, 1 + alpha, size=len(padre1[0]))
    alphas2 = np.random.uniform(-alpha, 1 + alpha, size=len(padre1[0]))
    hijo1[1] = alphas1 * padre1[1] + (1 - alphas1) * padre2[1]
    hijo1[1] /= np.sum(np.abs(hijo1[1]))
    hijo2[1] = (1 - alphas2) * padre1[1] + alphas2 * padre2[1]
    hijo2[1] /= np.sum(np.abs(hijo2[1]))
    return hijo1, hijo2

def predict(goat, Xtest):
    goat[1] = goat[1] * goat[0] / np.sum(goat[1] * goat[0])
    probs = np.dot(Xtest.astype(np.float64), goat[1, :].T.astype(np.float64))
    y_pred = (probs >= 0.5).astype(int)
    return y_pred

# # Adaptive mutation rate based on generation
# def adaptive_mutation_rate(generation, num_generations, initial_rate=0.1, final_rate=0.01):
#     return initial_rate - ((initial_rate - final_rate) * (generation / num_generations))

# # Dynamic tournament size to increase selection pressure
# def dynamic_tournament_size(generation, num_generations, initial_size=2, final_size=5):
#     return int(initial_size + ((final_size - initial_size) * (generation / num_generations)))
def dynamic_mutation_rate(start_time, max_time, min_rate=0.01, max_rate=0.1):
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        return min_rate  # Si el tiempo máximo se ha alcanzado, devolver la tasa mínima
    # Disminuir la tasa de mutación con el tiempo, pero nunca bajar de `min_rate`
    rate = max(min_rate, max_rate * (1 - elapsed_time / max_time))
    return rate

def dynamic_tournament_size(start_time, max_time, max_size=10):
    elapsed_time = time.time() - start_time
    
    # Si el tiempo máximo se ha alcanzado, devolver el tamaño mínimo
    if elapsed_time >= max_time:
        return 2  # Tamaño mínimo
    
    # Calcular el tamaño del torneo como una función del tiempo transcurrido
    size = 2 + int((elapsed_time / max_time) * (max_size - 2))  # Aumenta de 2 a max_size
    size = min(size, max_size)  # Asegurarse de no exceder max_size
    
    return size


def main():
    global interrupted
    if len(sys.argv) != 3:
        print("Uso: python main.py train.csv test.csv")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_df, test_df = load_data(train_file, test_file)
    Xtest = test_df
    Ytrain = train_df.iloc[:, -1]
    Xtrain = train_df.iloc[:, :-1]

    gen_actual = 0
    goat = np.zeros(Xtrain.shape[1])
    poblacion = generate_population(2000, Xtrain.shape[1])
    fit_max = 0
    fitnessPob = 0
    rng = np.random.default_rng()
    start_time = time.time()
    total_time = 60 * 60 * 4 # 4 horas en segundos
    
    while not interrupted:
        elapsed_time = time.time() - start_time  # Tiempo transcurrido
        # Fitness de la población actual
        fitnessPob = fitness_poblacion(poblacion, Xtrain, Ytrain, umbral=0.5, metrica='f1')
        
        if fit_max < fitness(poblacion[np.argmax(fitnessPob)], Xtrain, Ytrain):
            goat = poblacion[np.argmax(fitnessPob)]
            fit_max = fitness(poblacion[np.argmax(fitnessPob)], Xtrain, Ytrain)

        # Ajuste dinámico del tamaño del torneo y elitismo basado en el tiempo
        if elapsed_time < total_time * 0.33:  # Primer tercio (más exploración)
            num_elites = 1  # Poco elitismo, menor preservación de individuos
        elif elapsed_time < total_time * 0.66:  # Segundo tercio (balance)
            num_elites = 2  # Un poco más de elitismo
        else:  # Último tercio (más explotación)
            num_elites = 3  # Más elitismo para conservar los mejores individuos
        # Aumentar elitismo para mantener los mejores individuos
        mutation_rate = dynamic_mutation_rate(start_time, total_time)
        tournament_size = dynamic_tournament_size(start_time, total_time)  

        # Selección
        seleccionados = seleccion_torneo(fitnessPob, poblacion, tournament_size)

        # Reproducción y mutación
        hijos = []
        indices = np.arange(len(seleccionados))
        rng.shuffle(indices)
        for i in range(0, len(seleccionados), 2):
            padre1, padre2 = indices[i], indices[i + 1]
            hijo1, hijo2 = crossover(seleccionados[padre1], seleccionados[padre2])
            hijos.append(hijo1)
            hijos.append(hijo2)

        for hijo in hijos:
            mutacion_gaussiana_normalizada(hijo, prob_mutacion=mutation_rate)

        # Elitismo: reemplazar los peores con los mejores
        best_indices = np.argsort(fitnessPob)[-num_elites:]
        worst_indices = np.argsort(fitnessPob)[:num_elites]
        for i in range(num_elites):
            hijos[worst_indices[i]] = poblacion[best_indices[i]]

        poblacion = np.array(hijos)
        gen_actual += 1

    Ypred = predict(goat, Xtest)
    print(json.dumps(Ypred.tolist()))


if __name__ == "__main__":
    main()

