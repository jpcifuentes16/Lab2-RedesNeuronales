import numpy as np
from functools import reduce


# Función sigmoide
def sigmoid(mat):
    a = [(1 / (1 + np.exp(-x))) for x in mat]
    return np.asarray(a).reshape(mat.shape)

# FEED FORWARD
# Encuentra las matrices de activacion de cada neurona
def feed_forward(arr_thetas, X):

    # 2.1
    mat_list = [np.asarray(X)]

    # 2.2 
    for i in range(len(arr_thetas)):

        mat_list.append(
            # Aplicar la función sigmoide sobre z^i para obtener el siguiente arrelgo de a's
            sigmoid(
                # Multiplicación de matrices (a * theta transpuesta) = z^i
                np.matmul(
                    np.hstack((
                        # IMPORTANTE: Bias
                        np.ones(len(X)).reshape(len(X), 1),
                        mat_list[i]
                    )), arr_thetas[i].T
                )
            )            
        )
    return mat_list



# Algoritmo de back propagation
# X son las entradas de la red
# Y valor real de la prediccion
# flat_thetas son los pesos
# shapes las formas de cada capa
def back_propagation(flat_thetas, shapes, X, Y):
    m, capas = len(X), len(shapes) + 1
    thetas = inflate_matrixes(flat_thetas, shapes)

    # 2.2
    a = feed_forward(thetas, X)

    # 2.4
    deltas = [*range(capas - 1), a[-1] - Y]
    # 2.4
    for i in range(capas - 2, 0, -1):
        deltas[i] =  (deltas[i + 1] @ np.delete((thetas[i]), 0, 1)) * (a[i] * (1 - a[i]))

    # 2.5
    deltasFinal = []
    for i in range(capas - 1):
        deltasFinal.append(
            (
                deltas[i + 1].T @
                np.hstack((
                    np.ones(len(a[i])).reshape(len(a[i]), 1),
                    a[i]
                ))
            ) / m
        )

    deltasFinal = np.asarray(deltasFinal)

    # Paso 3
    return flatten_list_of_arrays(
        deltasFinal
    )
# Obtiene el costo entre la predicción y la respuesta
def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )

    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)
    
# Aplasta las matrices? 
flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

# Infla las matrices
def inflate_matrixes(flat_thetas, shapes):
    capas = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(capas, dtype=int)

    for i in range(capas - 1):
        steps[i + 1] = steps[i] + sizes[i]

    return [
        flat_thetas[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(capas - 1)
    ]

