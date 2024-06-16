import numpy as np
import pandas as pd

def expand(*dicts, keys, key=None):
    n = len(keys)
    key_to_index = {k: i for i, k in enumerate(keys)}
    column_index = {k: i for i, k in enumerate(dicts[0].keys())}

    if key:
        if key not in column_index:
            raise ValueError(f"Key '{key}' not found in dictionaries")

        row = np.zeros(n, dtype=np.float16)
        row_dict = {}
        for key2 in keys:
            value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
            row[key_to_index[key2]] = value
            row_dict[key2] = value
        return row, row_dict
    else:
        state_matrix = np.zeros((len(dicts[0]), n), dtype=np.float16)
        state_dict = {}
        for idx, key in enumerate(dicts[0]):
            row_dict = {}
            for key2 in keys:
                value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
                state_matrix[idx, key_to_index[key2]] = value
                row_dict[key2] = value
            state_dict[key] = row_dict
        return state_matrix, state_dict

def marginalize_row(dicts, index):
    new_dicts = []
    
    for d in dicts:
        transformed_dict = {}
        
        for key, value_dict in d.items():
            new_key = key[:index] + key[index + 1:]
            if new_key not in transformed_dict:
                transformed_dict[new_key] = {k: [] for k in value_dict}
            for k, v in value_dict.items():
                transformed_dict[new_key][k].append(v)
        
        for new_key, inner_dict in transformed_dict.items():
            for k, v_list in inner_dict.items():
                transformed_dict[new_key][k] = sum(v_list) / len(v_list)
        
        new_dicts.append(transformed_dict)
    
    return new_dicts

def obtener_cadena_valores(opcion):
    def backtrack(index, combinacion):
        if index == len(opcion):
            if '/' in combinacion:  
                combinaciones.append(combinacion)
            return
        backtrack(index + 1, combinacion + opcion[index])
        backtrack(index + 1, combinacion + '0')

    combinaciones = []
    
    backtrack(0, '')
    combinacionesOP = combinaciones[::-1]  # Invertimos la lista una vez al final
    return combinaciones, combinacionesOP

def getDistribution(distribution_str, dicts, key = None):
    # Generar la nueva clave para la expansión
    new_keys = [c for c in distribution_str.split('/')[0] if c != '0']

    # Crear mapeo dinámico basado en las claves originales
    dict_mapping = {chr(65 + i): i for i in range(len(dicts))}
    
    # Seleccionar los diccionarios correspondientes a las nuevas claves
    selected_dicts = [dicts[dict_mapping[key]] for key in new_keys]

    # Identificar índices que deben ser marginalizados
    marginalize_indices = [i for i, c in enumerate(distribution_str.split('/')[1]) if c == '0']
    
    # Realizar la marginalización
    for index in sorted(marginalize_indices, reverse=True):
        selected_dicts = marginalize_row(selected_dicts, index)
    
    # Expandir las matrices marginalizadas
    if(key):
        new_key = ''.join([key[dict_mapping[chr(65 + i)]] for i in range(len(dicts)) if chr(65 + i) in new_keys])
        if new_key not in selected_dicts[0]:
            raise ValueError(f"Key '{new_key}' not found in dictionaries")
        result, rdict = expand(*selected_dicts, keys=selected_dicts[0].keys(), key=new_key)
    else:
        result, rdict = expand(*selected_dicts, keys=selected_dicts[0].keys())

    rdict = {distribution_str.split('/')[1]: rdict}

    return result, rdict

def calcular_resultado(dict1, dict2, keys, key):
    resultado = {}
    resultadoF = {}

    new_keys1 = [c for c in next(iter(dict1.keys())) if c != '0']
    new_keys2 = [c for c in next(iter(dict2.keys())) if c != '0']

    dict3 = {chr(65 + i): i for i in range(len(new_keys1 + new_keys2))}

    all_letters = ''.join(dict3.keys()) + ' = ' + key

    for key in keys:
        selected_keys1 = [key[dict3[let]] for let in new_keys1]
        selected_keys2 = [key[dict3[let]] for let in new_keys2]

        # Form the actual keys to access values in dict1 and dict2
        key1 = ''.join(selected_keys1)
        key2 = ''.join(selected_keys2)

        value1 = next(iter(dict1.values()))[key1]
        value2 = next(iter(dict2.values()))[key2]

        # Calculate the result by multiplying values from dict1 and dict2
        result = value1 * value2
        
        # Store the result in the final dictionary with the current key
        resultado[key] = result

    resultadoF[all_letters] = resultado

    return resultadoF, all_letters

def calculate_emd(dist1, dist2, distance_matrix):
    supply = dist1.copy()
    demand = dist2.copy()
    total_cost = 0

    # Mientras haya suministro y demanda, mover la tierra
    while np.any(supply > 0) and np.any(demand > 0):
        i = np.argmax(supply)
        j = np.argmax(demand)

        # Cantidad a mover es el mínimo entre el suministro disponible y la demanda requerida
        amount = min(supply[i], demand[j])
        supply[i] -= amount
        demand[j] -= amount

        # Costo es cantidad movida por la distancia entre los bins
        total_cost += amount * distance_matrix[i, j]

    return total_cost

def hamming_distance(bin1, bin2):
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))


def fuerza_bruta(*dicts, key):
    combinaciones, combinacionesOP = obtener_cadena_valores("ABC/ABC")
    menor = 100

    for combinacion, combinacion_op in zip(combinaciones, combinacionesOP):
        result1, d1 = getDistribution(combinacion, dicts, key=key)
        
        result2, d2 = getDistribution(combinacion_op, dicts, key=key)
        
        rx, letters = calcular_resultado(d1, d2, Af.keys(), key=key)

        pdrx = pd.DataFrame.from_dict(rx).transpose()
        
        ordic2 = {}
        original, ordic = expand(dicts, keys=Af.keys(), key=key)
        ordic2[letters, ' = ', key] = ordic

        pdox = pd.DataFrame.from_dict(ordic2).transpose()
        print("\n")
        print(pdox)

        values1 = pdrx.iloc[0].values
        values2 = pdox.iloc[0].values

        keys = ['000', '100', '010', '110', '001', '101', '011', '111']

        distance_matrix = np.array([[hamming_distance(k1, k2) for k1 in keys] for k2 in keys], dtype=float)

        # Calcular la distancia EMD
        emd_distance = calculate_emd(values1, values2, distance_matrix)
        print(f"La distancia EMD es: {emd_distance}")

        if emd_distance < menor:
            menor = emd_distance
    
    print("Menor EMD: ", menor)


Af = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 0, '1': 1},
    '110': {'0': 0, '1': 1},
    '001': {'0': 0, '1': 1},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 0, '1': 1}
}

Bf = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 1, '1': 0},
    '010': {'0': 1, '1': 0},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 1, '1': 0},
    '111': {'0': 0, '1': 1}
}

Cf = {
    '000': {'0': 1, '1': 0},
    '100': {'0': 0, '1': 1},
    '010': {'0': 0, '1': 1},
    '110': {'0': 1, '1': 0},
    '001': {'0': 1, '1': 0},
    '101': {'0': 0, '1': 1},
    '011': {'0': 0, '1': 1},
    '111': {'0': 1, '1': 0}
}

# EJEMPLO
key = '100'
distribution_str = "A0C/0BC"
result1, d1= getDistribution(distribution_str, [Af, Bf, Cf], key=key)
print(d1)

distribution_str = "0B0/A00"
result2, d2 = getDistribution(distribution_str, [Af, Bf, Cf], key=key)
print(d2)

rx, letters = calcular_resultado(d1, d2, Af.keys(), key = key)
print("\n")

pdrx = pd.DataFrame.from_dict(rx).transpose()
print(pdrx)

ordic2 = {}
original, ordic = expand(Af, Bf, Cf, keys=Af.keys(), key = key)
ordic2[letters] = ordic

pdox = pd.DataFrame.from_dict(ordic2).transpose()
print("\n")
print(pdox)

values1 = pdrx.iloc[0].values
values2 = pdox.iloc[0].values

keys = ['000', '100', '010', '110', '001', '101', '011', '111']

distance_matrix = np.array([[hamming_distance(k1, k2) for k1 in keys] for k2 in keys], dtype=float)

# Calcular la distancia EMD
emd_distance = calculate_emd(values1, values2, distance_matrix)

print("\n")
print(f"La distancia EMD es: {emd_distance}")