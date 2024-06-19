import numpy as np
import pandas as pd

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
        state_dict = {}
        for idx, key in enumerate(dicts[0]):
            row_dict = {}
            for key2 in column_index:
                value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
                state_matrix[idx, column_index[key2]] = value
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

def marginalize_column(dict, index):
    transformed_dict = {}
    for key, value_dict in dict.items():
        if key not in transformed_dict:
            transformed_dict[key] = {}
        for k, v in value_dict.items():
            new_key = k[:index] + k[index + 1:]
            if new_key not in transformed_dict[key]:
                transformed_dict[key][new_key] = v
            else:
                transformed_dict[key][new_key] += v
    return transformed_dict

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
    combinacionesOP = combinaciones[::-1]
    return combinaciones[1:-1], combinacionesOP[1:-1]

def getDistribution(distribution_str, dicts, key, cache):
    if (distribution_str, key) in cache:
        return cache[(distribution_str, key)]
    
    new_keys = [c for c in distribution_str.split('/')[0] if c != '0']
    new_keys_cols = [c for c in distribution_str.split('/')[1] if c != '0']
    cadenas = distribution_str.split('/')
    cadenas = '/'.join(cadenas).replace('000', '∅')
    dict_mapping = {chr(65 + i): i for i in range(len(dicts))}
    selected_dicts = [dicts[dict_mapping[key]] for key in new_keys]
    marginalize_indices = [i for i, c in enumerate(distribution_str.split('/')[1]) if c == '0']

    if new_keys and not new_keys_cols:
        key = None
        marginalize_indices = []

    for index in sorted(marginalize_indices, reverse=True):
        selected_dicts = marginalize_row(selected_dicts, index)

    if key:
        new_key = ''.join([key[dict_mapping[chr(65 + i)]] for i in range(len(dicts)) if chr(65 + i) in new_keys_cols])
        if not selected_dicts:
            result = [1]
            rdict = {cadenas.replace('0', '') + " = " + new_key: {new_key: 1}}
        else:
            if new_key not in selected_dicts[0]:
                raise ValueError(f"Key '{new_key}' not found in dictionaries")
            result, rdict = expand(*selected_dicts, keys=selected_dicts[0].keys(), key=new_key)
            rdict = {cadenas.replace('0', '') + " = " + new_key: rdict}
    else:
        result, rdict = expand(*selected_dicts, keys=selected_dicts[0].keys())
        column_index = {format(i, '0' + str(len(selected_dicts)) + 'b')[::-1]: 0 for i in range(2 ** len(selected_dicts))}
        for key in rdict:
            for key2 in column_index:
                column_index[key2] += rdict[key][key2] / len(rdict.keys())
        rdict = {cadenas.replace('0', ''): column_index}

    cache[(distribution_str, key)] = (result, rdict)
    return result, rdict

def calcular_resultado(dict1, dict2, keys, key, cache):
    def dict_to_hashable(d):
        return frozenset((k, frozenset(v.items())) for k, v in d.items())

    cache_key = (dict_to_hashable(dict1), dict_to_hashable(dict2), key)
    if cache_key in cache:
        return cache[cache_key]
    
    resultado = {}
    resultadoF = {}
    new_keys1 = [c for c in next(iter(dict1.keys())).split('/')[0] if c != '0']
    new_keys2 = [c for c in next(iter(dict2.keys())).split('/')[0] if c != '0']

    if '∅' in new_keys1:
        new_keys1 = []
    if '∅' in new_keys2:
        new_keys2 = []

    dict3 = {chr(65 + i): i for i in range(len(new_keys1 + new_keys2))}
    all_letters = ''.join(dict3.keys()) + ' = ' + key

    if not new_keys1 and new_keys2:
        for key in keys:
            selected_keys2 = [key[dict3[let]] for let in new_keys2]
            key2 = ''.join(selected_keys2)
            value2 = next(iter(dict2.values()))[key2]
            resultado[key] = value2
        resultadoF[all_letters] = resultado
        cache[cache_key] = (resultadoF, all_letters)
        return resultadoF, all_letters
    elif not new_keys2 and new_keys1:
        for key in keys:
            selected_keys1 = [key[dict3[let]] for let in new_keys1]
            key1 = ''.join(selected_keys1)
            value1 = next(iter(dict1.values()))[key1]
            resultado[key] = value1
        resultadoF[all_letters] = resultado
        cache[cache_key] = (resultadoF, all_letters)
        return resultadoF, all_letters

    for key in keys:
        selected_keys1 = [key[dict3[let]] for let in new_keys1 if let != '∅']
        selected_keys2 = [key[dict3[let]] for let in new_keys2 if let != '∅']
        key1 = ''.join(selected_keys1)
        key2 = ''.join(selected_keys2)
        value1 = next(iter(dict1.values())).get(key1, 1)  # Asumir 1 si la clave no existe
        value2 = next(iter(dict2.values())).get(key2, 1)  # Asumir 1 si la clave no existe
        result = value1 * value2
        resultado[key] = result

    resultadoF[all_letters] = resultado
    cache[cache_key] = (resultadoF, all_letters)
    return resultadoF, all_letters

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
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

def bottom_up(*dicts, key):
    combinaciones, combinacionesOP = obtener_cadena_valores("ABC/ABC")
    menor = np.inf
    best_combination = ""
    best_dict = None

    keys = dicts[0].keys()
    n = len(keys)
    distance_matrix = np.zeros((n, n))

    ordic2 = {}
    original, ordic = expand(dicts[0], dicts[1], dicts[2], keys=dicts[0].keys(), key=key)
    ordic2[f"ABC = ", key] = ordic

    pdox = pd.DataFrame.from_dict(ordic2).transpose()
    print("\nOriginal:\n\n", pdox) 

    values2 = pdox.iloc[0].values

    for i, keyy in enumerate(keys):
        for j, key2 in enumerate(keys):
            distance_matrix[i, j] = hamming_distance(keyy, key2)

    cache = {}

    for combinacion, combinacion_op in zip(combinaciones, combinacionesOP):
        result1, d1 = getDistribution(combinacion, dicts, key=key, cache=cache)
        result2, d2 = getDistribution(combinacion_op, dicts, key=key, cache=cache)
        rx, letters = calcular_resultado(d1, d2, dicts[0].keys(), key=key, cache=cache)
        pdrx = pd.DataFrame.from_dict(rx).transpose()
        values1 = pdrx.iloc[0].values
        emd_distance = calculate_emd(values1, values2, distance_matrix)

        if emd_distance < menor:
            menor = emd_distance
            best_combination = combinacion + " * " + combinacion_op
            best_dict = pdrx
    
    best_combination = best_combination.replace('000', '∅').replace('0', '')
    print("\n\nMenor EMD: ", menor)
    print("Mejor combinación: ", best_combination)
    print("Mejor distribución:\n\n", best_dict)

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

#bottom_up(Af, Bf, Cf, key='001')