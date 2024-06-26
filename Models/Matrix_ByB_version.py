import numpy as np
import heapq
import itertools
import string
from collections import deque
from collections import defaultdict


class State:
    def __init__(self, peso, tabla=None, solucion=None, cortes=None, hijosState=None):
        self.peso = float(peso)
        self.tabla = (
            tabla or []
        )  # Manejar adecuadamente la tabla        self.solucion = solucion or False  # Valor predeterminado: False si no se proporciona
        self.cortes = (
            cortes or []
        )  # Valor predeterminado: lista vacía si no se proporciona
        self.solucion = (
            solucion or False
        )  # Valor predeterminado: False si no se proporciona
        self.hijosState = hijosState or []
    def __lt__(self, other):
        return self.peso < other.peso

    def __str__(self):
        # Representación legible de la clase
        return f"Problema de Optimización:\nPeso: {self.peso}\nTabla:\n{self.tabla}\nSolución: {self.solucion}\nCortes: {self.cortes}"


class MatrizBipartita:
    def __init__(self, matriz):
        self.matriz_original = matriz
        self.matriz_binaria = self.convertir_a_matriz_binaria(matriz)

    def convertir_a_matriz_binaria(self, matriz):
        matriz_binaria = [[1 if valor != 0 else 0 for valor in fila] for fila in matriz]
        return matriz_binaria

    def actualizar_matriz(self, fila, columna, valor):
        self.matriz_original[fila][columna] = valor
        self.matriz_binaria = self.convertir_a_matriz_binaria(self.matriz_original)

    def __str__(self):
        # Para imprimir la matriz binaria
        return "\n".join([" ".join(map(str, fila)) for fila in self.matriz_binaria])


class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, state):
        heapq.heappush(self.heap, (state.peso, state))  # Usamos peso como prioridad

    def pop(self):
        return heapq.heappop(self.heap)[1]  # Devolvemos solo el objeto State

    def peek(self):
        return self.heap[0][1]

    def is_empty(self):
        return len(self.heap) == 0

    def get_all_states(self):
        return [state for _, state in self.heap]

    class PriorityQueueWrapper:
        def __init__(self):
            self.pq = PriorityQueue()

        def is_empty(self):
            return self.pq.empty()

        def put(self, state):
            self.pq.put(state)

        def get(self):
            return self.pq.get()

        def peek(self):
            return self.pq.queue[0] if not self.is_empty() else None

        def get_all_states(self):
            return list(self.pq.queue)


def isBipartite(adj_matrix):
    n = len(adj_matrix)
    color = [-1] * n  # -1 means uncolored

    for start_node in range(n):
        if color[start_node] == -1:  # Node is not colored
            queue = deque([start_node])
            color[start_node] = 0  # Start coloring with 0

            while queue:
                current = queue.popleft()
                current_color = color[current]
                next_color = 1 - current_color

                for neighbor in range(n):
                    if adj_matrix[current][neighbor] == 1:  # There is an edge
                        if color[neighbor] == -1:
                            color[neighbor] = next_color
                            queue.append(neighbor)
                        elif color[neighbor] == current_color:
                            return False
    return True


def connectedComponents(adj_matrix):
    def dfs(node_id, visited, component_id):
        stack = [node_id]
        visited[node_id] = component_id

        while stack:
            node = stack.pop()
            for neighbor in range(len(adj_matrix)):
                if adj_matrix[node][neighbor] == 1 and visited[neighbor] == -1:
                    visited[neighbor] = component_id
                    stack.append(neighbor)

    n = len(adj_matrix)
    visited = [-1] * n  # -1 means unvisited
    component_count = 0

    for node in range(n):
        if visited[node] == -1:  # Node is not visited
            component_count += 1
            dfs(node, visited, component_count)

    return component_count, visited  # Return number of components and their assignments


def condensar_y_restaurar_tabla(Bf, eliminar_col):
    def procesar_estado(estado, futuros):
        estado_modificado = estado[:eliminar_col] + estado[eliminar_col + 1 :]
        estado_original[estado_modificado].append((estado, futuros))
        for futuro, probabilidad in futuros.items():
            agrupaciones[estado_modificado][futuro].append(probabilidad)

    def promediar_probabilidades(futuros):
        return {
            futuro: sum(probabilidades) / len(probabilidades)
            for futuro, probabilidades in futuros.items()
        }

    def restaurar_tabla():
        tabla_restaurada = {}
        for estado, futuros in Bf.items():
            estado_modificado = estado[:eliminar_col] + estado[eliminar_col + 1 :]
            if estado_modificado in nueva_tabla:
                tabla_restaurada[estado] = nueva_tabla[estado_modificado]
        return tabla_restaurada

    estado_original = defaultdict(list)
    agrupaciones = defaultdict(lambda: defaultdict(list))

    for estado, futuros in Bf.items():
        procesar_estado(estado, futuros)

    nueva_tabla = {
        estado: promediar_probabilidades(futuros)
        for estado, futuros in agrupaciones.items()
    }

    return restaurar_tabla()


def expand(*dicts, keys, key=None):
    n = 2 ** len(dicts)
    key_to_index = {k: i for i, k in enumerate(keys)}
    num_combinaciones = 2 ** len(dicts)

    column_index = {
        format(i, "0" + str(len(dicts)) + "b")[::-1]: idx
        for idx, i in enumerate(range(num_combinaciones))
    }

    if key:
        if key not in key_to_index:
            raise ValueError(f"Key '{key}' not found in dictionaries")
        row = np.zeros(n, dtype=np.float16)
        row_dict = {}
        for key2 in column_index:
            value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
            row[column_index[key2]] = value
            row_dict[key2] = value
        return row, row_dict
    else:
        state_matrix = np.zeros((len(dicts[0]), n), dtype=np.float16)
        for idx, key in enumerate(dicts[0]):
            row_dict = {}
            for key2 in column_index:
                value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
                state_matrix[idx, column_index[key2]] = value
                row_dict[key2] = value
        return state_matrix


def build_distance_matrix(states):
    n = len(states)
    distance_matrix = np.zeros((n, n), dtype=np.float64)

    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            distance_matrix[i, j] = hamming_distance(state1, state2)

    return distance_matrix


def calculate_emd(dist1, dist2, distance_matrix):
    supply = np.array(dist1, dtype=np.float64)
    demand = np.array(dist2, dtype=np.float64)
    total_cost = 0.0
    flow = np.zeros((len(supply), len(demand)))

    while np.any(supply > 0) and np.any(demand > 0):
        i = np.argmax(supply)
        j = np.argmax(demand)
        amount = min(supply[i], demand[j])
        supply[i] -= amount
        demand[j] -= amount
        flow[i, j] = amount
        total_cost += amount * distance_matrix[i, j]

    return total_cost


def hamming_distance(bin1, bin2):
    if len(bin1) != len(bin2):
        raise ValueError("Los vectores deben tener la misma longitud")
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def generar_combinaciones_pares(num_letras):
    letras = string.ascii_uppercase[:num_letras]
    combinaciones = list(itertools.permutations(letras, 2))  # Convertir a lista
    return combinaciones


A = {
    "000": {"0": 1, "1": 0},
    "100": {"0": 1, "1": 0},
    "010": {"0": 0, "1": 1},
    "110": {"0": 0, "1": 1},
    "001": {"0": 0, "1": 1},
    "101": {"0": 0, "1": 1},
    "011": {"0": 0, "1": 1},
    "111": {"0": 0, "1": 1},
}

B = {
    "000": {"0": 1, "1": 0},
    "100": {"0": 1, "1": 0},
    "010": {"0": 1, "1": 0},
    "110": {"0": 1, "1": 0},
    "001": {"0": 1, "1": 0},
    "101": {"0": 0, "1": 1},
    "011": {"0": 1, "1": 0},
    "111": {"0": 0, "1": 1},
}

C = {
    "000": {"0": 1, "1": 0},
    "100": {"0": 0, "1": 1},
    "010": {"0": 0, "1": 1},
    "110": {"0": 1, "1": 0},
    "001": {"0": 1, "1": 0},
    "101": {"0": 0, "1": 1},
    "011": {"0": 0, "1": 1},
    "111": {"0": 1, "1": 0},
}


def generar_combinaciones_pares(num_letras):
    letras = string.ascii_uppercase[:num_letras]
    combinaciones = list(itertools.permutations(letras, 2))  # Convertir a lista
    return combinaciones


def create_statesiniciales(*dicts, key):
    pq = PriorityQueue()
    nLetra = len(dicts)
    matrixad = np.zeros((nLetra, nLetra))
    tabla_original = expand(*dicts, keys=dicts[0].keys())
    distance_matrix = build_distance_matrix(
        list(dicts[0].keys())
    )  # Usamos las claves de la primera tabla
    num = list(dicts[0].keys())
    posicion = num.index(key)
    orden = generar_combinaciones_pares(nLetra)
    letra_a_indice = {
        letra: idx for idx, letra in enumerate(string.ascii_uppercase[:nLetra])
    }

    for par in orden:
        tabla_a_modificar = dicts[letra_a_indice[par[0]]]
        col_a_eliminar = letra_a_indice[par[1]]
        tabla_modificada = condensar_y_restaurar_tabla(
            tabla_a_modificar, col_a_eliminar
        )

        # Crear una nueva lista de tablas para la expansión, reemplazando la tabla modificada
        tablas_para_expandir = [
            tabla_modificada if i == letra_a_indice[par[0]] else dicts[i]
            for i in range(nLetra)
        ]

        tabla_expandido = expand(*tablas_para_expandir, keys=dicts[0].keys())

        filaO = tabla_original[posicion]
        filaA = tabla_expandido[posicion]

        emd_exacto = calculate_emd(filaO, filaA, distance_matrix)

        if emd_exacto == 0.0:
            dicts = tablas_para_expandir = [
                tabla_modificada if i == letra_a_indice[par[0]] else dicts[i]
                for i in range(nLetra)
            ]

        matrixad[letra_a_indice[par[1]], letra_a_indice[par[0]]] = emd_exacto

        matriz_objeto = MatrizBipartita(matrixad)

        state = State(peso=emd_exacto, tabla=dicts, solucion=None, cortes=[par])
        # print(state.cortes)
        pq.push(state)

    return pq, matriz_objeto  # Devolver la cola de prioridad y la matriz de objetos


# Define tus funciones y clases auxiliares según sea necesario
def Branch_and_Bound(*dicts, key):
    pq, mat = create_statesiniciales(*dicts, key=key)
    cotaGlobal = np.inf  # cota global inicializada en infinito

    distance_matrix = build_distance_matrix(list(dicts[0].keys()))  # Matriz de distancias

    nLetra = len(dicts)  # Número de letras

    orden = generar_combinaciones_pares(nLetra)  # Generar combinaciones de pares

    num = list(dicts[0].keys())  # Lista de claves
    posicion = num.index(key)   # Posición de la clave que entra

    letra_a_indice = {
        letra: idx for idx, letra in enumerate(string.ascii_uppercase[:nLetra])  # Diccionario de letras a índices
    }
    
    lista = []
    todos_los_cortes = []
    pq_copy = pq.get_all_states()  # Copia de la cola de prioridad

    soluciones = []  # Lista de soluciones encontradas

    for i in pq_copy:
        lista.append(i.cortes[0])  # Lista de cortes en general y orden
    
    while not pq.is_empty():  # Mientras la cola de prioridad no esté vacía

        current_state = pq.peek()  # Obtener el estado actual de la cola de prioridad (mayor prioridad)

        if current_state.solucion:  # Si es una solución
            soluciones.append((current_state.peso, current_state.cortes))  # Agregar a la lista de soluciones

        orden = [x for x in lista if x not in current_state.cortes]  # Orden de los cortes

        for par in orden:  # Para cada par en el orden

            tabla_a_modificar = current_state.tabla[letra_a_indice[par[0]]]  # Tabla a modificar
            col_a_eliminar = letra_a_indice[par[1]]
            tabla_modificada = condensar_y_restaurar_tabla(tabla_a_modificar, col_a_eliminar)

            tablas_para_expandir = [
                tabla_modificada if i == letra_a_indice[par[0]] else current_state.tabla[i]
                for i in range(nLetra)
            ]

            tabla_expandido = expand(*tablas_para_expandir, keys=dicts[0].keys())

            tablas_para_expandir_o = [
                current_state.tabla[i] if i == letra_a_indice[par[0]] else current_state.tabla[i]
                for i in range(nLetra)
            ]
            tabla_original = expand(*tablas_para_expandir_o, keys=dicts[0].keys())

            filaO = tabla_original[posicion]
            filaA = tabla_expandido[posicion]

            emd_exacto = calculate_emd(filaO, filaA, distance_matrix)
            mat.actualizar_matriz(letra_a_indice[par[1]], letra_a_indice[par[0]], 0)

            ncomponents, n = connectedComponents(mat.matriz_binaria)

            if ncomponents == 2:
                if emd_exacto < cotaGlobal:
                    cotaGlobal = emd_exacto
                    soluciones.append((current_state.peso, current_state.cortes[:]))
                    current_state.tabla = tablas_para_expandir_o
                    mat.actualizar_matriz(letra_a_indice[par[1]], letra_a_indice[par[0]], emd_exacto)
                    print(soluciones)
                    
            else:
                current_state.peso += emd_exacto 
                current_state.tabla = tablas_para_expandir
                mat.actualizar_matriz(letra_a_indice[par[1]], letra_a_indice[par[0]], emd_exacto)
            

        pq.pop()
        current_state.cortes = [par for par in current_state.cortes if par in orden]

    return soluciones


# Imprimir contenido de la PriorityQueue
pq = Branch_and_Bound(A, B, C, key="100")
