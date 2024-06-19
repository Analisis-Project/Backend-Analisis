import numpy as np
import heapq
import itertools
import string
from collections import defaultdict
import Matrix_PD_version as mpd

class State:
    def __init__(self, peso, tabla, solucion=None, cortes=None):
        self.peso = float(peso)
        self.tabla = np.array(tabla)  # Convertir la tabla a un array NumPy
        self.solucion = solucion or False  # Valor predeterminado: False si no se proporciona
        self.cortes = cortes or []  # Valor predeterminado: lista vacía si no se proporciona
        
        # Validación básica (puedes agregar más según tus necesidades)
        if self.tabla.ndim != 2:
            raise ValueError("La tabla debe ser una matriz bidimensional.")
    
    def __lt__(self, other):
        return self.peso < other.peso

    def __str__(self):
        # Representación legible de la clase
        return f"Problema de Optimización:\nPeso: {self.peso}\nTabla:\n{self.tabla}\nSolución: {self.solucion}\nCortes: {self.cortes}"

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, state):
        heapq.heappush(self.heap, (state.peso, state))  # Usamos peso como prioridad

    def pop(self):
        return heapq.heappop(self.heap)[1]  # Devolvemos solo el objeto State

    def is_empty(self):
        return len(self.heap) == 0

def condensar_y_restaurar_tabla(Bf, eliminar_col):
    def procesar_estado(estado, futuros):
        estado_modificado = estado[:eliminar_col] + estado[eliminar_col+1:]
        estado_original[estado_modificado].append((estado, futuros))
        for futuro, probabilidad in futuros.items():
            agrupaciones[estado_modificado][futuro].append(probabilidad)

    def promediar_probabilidades(futuros):
        return {futuro: sum(probabilidades) / len(probabilidades) for futuro, probabilidades in futuros.items()}

    def restaurar_tabla():
        tabla_restaurada = {}
        for estado, futuros in Bf.items():
            estado_modificado = estado[:eliminar_col] + estado[eliminar_col+1:]
            if estado_modificado in nueva_tabla:
                tabla_restaurada[estado] = nueva_tabla[estado_modificado]
        return tabla_restaurada

    estado_original = defaultdict(list)
    agrupaciones = defaultdict(lambda: defaultdict(list))

    for estado, futuros in Bf.items():
        procesar_estado(estado, futuros)

    nueva_tabla = {estado: promediar_probabilidades(futuros) for estado, futuros in agrupaciones.items()}

    return restaurar_tabla()

def expand(*dicts, keys, key=None):
    n = 2**len(dicts)
    key_to_index = {k: i for i, k in enumerate(keys)}
    num_combinaciones = 2 ** len(dicts)

    column_index = {
        format(i, '0' + str(len(dicts)) + 'b')[::-1]: idx
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



# Función para obtener una fila específica del estado expandido
def obtener_fila_expandida(estado_expandido, estado_clave):
    if estado_clave not in estado_expandido:
        raise KeyError(f"El estado clave '{estado_clave}' no se encuentra en estado_expandido.")
    
    # Obtener la distribución asociada con el estado_clave
    distribucion = estado_expandido[estado_clave]

    # Convertir la distribución a un array de NumPy
    fila_expandida = np.array(list(distribucion.values()), dtype=np.float64)

    return fila_expandida
# Definir la tabla original
A = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 0, '1': 1},
    '110': {'0': 0, '1': 1},
    '001': {'0': 0, '1': 1},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 0, '1': 1}
}

B = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 1, '1': 0},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 1, '1': 0},
    '111': {'0': 0, '1': 1}
}

C = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 0, '1': 1},
    '010': {'0': 0, '1': 1},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 1, '1': 0}
}

def generar_combinaciones_pares(num_letras):
    letras = string.ascii_uppercase[:num_letras]
    combinaciones = list(itertools.permutations(letras, 2))  # Convertir a lista
    return combinaciones

def create_statesiniciales(*dicts, key):
    pq = PriorityQueue()
    nLetra = len(dicts)
    matrixad=np.zeros((nLetra,nLetra))
    tabla_original = expand(*dicts, keys=dicts[0].keys())
    distance_matrix = build_distance_matrix(list(dicts[0].keys()))  # Usamos las claves de la primera tabla
    num = list(dicts[0].keys())
    posicion = num.index(key)
    orden = generar_combinaciones_pares(nLetra)
    letra_a_indice = {letra: idx for idx, letra in enumerate(string.ascii_uppercase[:nLetra])}

    for par in orden:
        tabla_a_modificar = dicts[letra_a_indice[par[0]]]
        col_a_eliminar = letra_a_indice[par[1]]
        tabla_modificada = condensar_y_restaurar_tabla(tabla_a_modificar, col_a_eliminar)
        
        # Crear una nueva lista de tablas para la expansión, reemplazando la tabla modificada
        tablas_para_expandir = [
            tabla_modificada if i == letra_a_indice[par[0]] else dicts[i]
            for i in range(nLetra)
        ]

        tabla_expandido = expand(*tablas_para_expandir, keys=dicts[0].keys())
        
        # Obtener las filas expandidas
        filaO = tabla_original[posicion]
        filaA = tabla_expandido[posicion]
 
        emd_exacto = calculate_emd(filaO, filaA, distance_matrix)

        if emd_exacto == 0.0:
            dicts = getIndividualMatrixes(tabla_expandido, nLetra)
            print(dicts)
            

        matrixad[letra_a_indice[par[0]], letra_a_indice[par[1]]] = emd_exacto       
        state = State(peso=emd_exacto, tabla=tabla_expandido, solucion=None, cortes=[par])
        print(state)
        pq.push(state)
    
    return pq , matrixad




def Branch_and_Bound(*dicts, key):
    pq  , mat= create_statesiniciales(*dicts, key=key)
    #print(mat)
    cotaGlobal = np.inf
    tabla_original = expand(*dicts, keys=dicts[0].keys())
    distance_matrix = build_distance_matrix(list(dicts[0].keys()))

    while not pq.is_empty():
        current_state = pq.pop()

        if current_state.solucion:
            return current_state

        nLetra = len(dicts)
        letra_a_indice = {letra: idx for idx, letra in enumerate(string.ascii_uppercase[:nLetra])}
        num = list(dicts[0].keys())
        posicion = num.index(key)

        for par in generar_combinaciones_pares(nLetra):
            if par not in current_state.cortes:                    
                tabla_a_modificar = dicts[letra_a_indice[par[0]]]
                col_a_eliminar = letra_a_indice[par[1]]
                tabla_modificada = condensar_y_restaurar_tabla(tabla_a_modificar, col_a_eliminar)
                
                tablas_para_expandir = [
                    tabla_modificada if i == letra_a_indice[par[0]] else dicts[i]
                    for i in range(nLetra)
                ]

                tabla_expandido = expand(*tablas_para_expandir, keys=dicts[0].keys())
                filaO = tabla_original[posicion]
                filaA = tabla_expandido[posicion]
                emd_exacto = calculate_emd(filaO, filaA, distance_matrix)

                if emd_exacto < cotaGlobal:
                    new_state = State(peso=emd_exacto, tabla=tabla_expandido, cortes=current_state.cortes + [par])
                    pq.push(new_state)

    return pq

def getIndividualMatrixes(dict, column_indices):
    dicts = []
    for index in range(column_indices):
        aux_dict = dict
        for index2 in range(column_indices, 0, -1):
            if (index2 - 1) != index:
                aux_dict = mpd.marginalize_column(aux_dict, index2 - 1)
        dicts.append(aux_dict)
    return dicts


pq = Branch_and_Bound(A, B, C, key='100')

#ab=condensar_y_restaurar_tabla(B, 0)
#abc=condensar_y_restaurar_tabla(ab, 2)

#tabla =expand(A, abc, C, keys=A.keys())
#print(tabla)

