import numpy as np
from collections import deque

def es_bipartito(matriz):
    n = len(matriz)
    colores = [-1] * n  # -1: no coloreado, 0: color 0, 1: color 1

    for start in range(n):
        if colores[start] == -1:
            queue = deque([start])
            colores[start] = 0  # Asignar el primer color

            while queue:
                nodo = queue.popleft()
                for vecino in range(n):
                    if matriz[nodo, vecino] > 0:  # Hay una arista
                        if colores[vecino] == -1:  # No coloreado
                            colores[vecino] = 1 - colores[nodo]
                            queue.append(vecino)
                        elif colores[vecino] == colores[nodo]:  # Mismo color que el nodo actual
                            return False, []

    return True, colores

def contar_componentes(matriz):
    n = len(matriz)
    visitado = [False] * n

    def dfs(v):
        stack = [v]
        while stack:
            nodo = stack.pop()
            for vecino in range(n):
                if matriz[nodo, vecino] > 0 and not visitado[vecino]:
                    visitado[vecino] = True
                    stack.append(vecino)

    num_componentes = 0
    for i in range(n):
        if not visitado[i]:
            num_componentes += 1
            dfs(i)

    return num_componentes

# Matriz de adyacencia proporcionada
matriz_adyacencia = np.array([
    [0.0, 0.5, 0.5],
    [0.0, 0.0, 0],
    [0.5, 0.5, 0.0]
])

# Verificar si es bipartito
es_bip, colores = es_bipartito(matriz_adyacencia)
print(f"El grafo es bipartito: {es_bip}")
if es_bip:
    print(f"Colores asignados: {colores}")

# Contar componentes conexas
num_componentes = contar_componentes(matriz_adyacencia)
print(f"El número de componentes conexas es: {num_componentes}")
