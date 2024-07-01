import numpy as np
import pandas as pd
from collections import defaultdict

cache_marginalize_row = {}
cache_marginalize_column = {}

def dict_to_hashable(d):
    return frozenset((k, frozenset(v.items())) for k, v in d.items())

def expand(dicts, keys, key=None):
    n = 2 ** len(dicts)
    key_to_index = {k: i for i, k in enumerate(keys)}
    num_combinaciones = 2 ** len(dicts)

    column_index = {
        format(i, '0' + str(len(dicts)) + 'b')[::-1]: idx
        for idx, i in enumerate(range(num_combinaciones))
    }

    state_matrix = np.zeros((len(dicts[0]), n), dtype=np.float16)
    state_dict = {}
    
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
    
    for idx, key in enumerate(dicts[0]):
        row_dict = {}
        for key2 in column_index:
            value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
            state_matrix[idx, column_index[key2]] = value
            row_dict[key2] = value
        state_dict[key] = row_dict
    return state_matrix, state_dict

def marginalize_row(dicts, index):
    cache_key = (tuple(dict_to_hashable(d) for d in dicts), index)
    if cache_key in cache_marginalize_row:
        return cache_marginalize_row[cache_key]

    new_dicts = []
    for d in dicts:
        transformed_dict = defaultdict(lambda: defaultdict(list))
        for key, value_dict in d.items():
            new_key = key[:index] + key[index + 1:]
            for k, v in value_dict.items():
                transformed_dict[new_key][k].append(v)
        for new_key, inner_dict in transformed_dict.items():
            for k, v_list in inner_dict.items():
                transformed_dict[new_key][k] = sum(v_list) / len(v_list)
        new_dicts.append(transformed_dict)

    cache_marginalize_row[cache_key] = new_dicts
    return new_dicts

def marginalize_column(dict, index):
    cache_key = (dict_to_hashable(dict), index)
    if cache_key in cache_marginalize_column:
        return cache_marginalize_column[cache_key]

    transformed_dict = defaultdict(lambda: defaultdict(float))
    for key, value_dict in dict.items():
        for k, v in value_dict.items():
            new_key = k[:index] + k[index + 1:]
            transformed_dict[key][new_key] = transformed_dict[key].get(new_key, 0) + v

    cache_marginalize_column[cache_key] = transformed_dict
    return transformed_dict

def obtener_cadena_valores(opcion):
    combinaciones = []

    def backtrack(index, combinacion):
        if index == len(opcion):
            if '/' in combinacion:
                combinaciones.append(combinacion)
            return
        backtrack(index + 1, combinacion + opcion[index])
        backtrack(index + 1, combinacion + '0')

    backtrack(0, '')
    combinacionesOP = combinaciones[::-1]
    return combinaciones[1:-1], combinacionesOP[1:-1]

def getDistribution(distribution_str, dicts, key, cache, row_map, col_map, conditions, actual):
    if (distribution_str, key) in cache:
        return cache[(distribution_str, key)]
    
    new_keys = [c for c in distribution_str.split('/')[1] if c != '0']
    new_keys_cols = [c for c in distribution_str.split('/')[0] if c != '0']

    empty_col = ''.join(['0' for _ in col_map])
    empty_row = ''.join(['0' for _ in row_map])
    cadenas = distribution_str.split('/')
    cadenas[0] = cadenas[0].replace(empty_col, '∅')
    cadenas[1] = cadenas[1].replace(empty_row, '∅')
    cadenas = '/'.join(cadenas)
    
    selected_dicts = [dicts[col_map[key]] for key in new_keys_cols]
    selected_condition = {key: conditions[key] for key in new_keys_cols}
    marginalize_indices = [i for i, c in enumerate(distribution_str.split('/')[1]) if c == '0']

    if new_keys_cols and not new_keys:
        key = None
        marginalize_indices = []

    for index in sorted(marginalize_indices, reverse=True):
        selected_dicts = marginalize_row(selected_dicts, index)

    if key:
        new_key = ''.join([key[row_map[chr(65 + i)]] for i in range(10) if chr(65 + i) in new_keys])
        if not selected_dicts:
            result = [1]
            rdict = {cadenas.replace('0', ''): {new_key: 1}}
        else:
            aux = 0
            for i, value in enumerate(selected_condition.values()):
                max_valor = max(selected_dicts[i][new_key].values())
                maximo = [clave for clave, valor in selected_dicts[i][new_key].items() if valor == max_valor]
                if(value in maximo):
                    aux += 1
            
            if aux == len(selected_condition):
                nose = 2
            elif aux == 0:
                nose = 0
            else:
                nose = 1
            
            if nose >= actual:
                actual = nose
                if new_key not in selected_dicts[0]:
                    raise ValueError(f"Key '{new_key}' not found in dictionaries")
                result, rdict = expand(selected_dicts, keys=selected_dicts[0].keys(), key=new_key)
                rdict = {cadenas.replace('0', ''): rdict}
            else:
                cache[(distribution_str, key)] = (actual, None, None)
                return actual, None, None
    else:
        aux = 0
        for i, value in enumerate(selected_condition.values()):
            sum_0 = 0
            sum_1 = 0
            count = len(selected_dicts[i])
            for key in selected_dicts[i]:
                sum_0 += selected_dicts[i][key]['0']
                sum_1 += selected_dicts[i][key]['1']
                
            average_dict = {'0': sum_0 / count, '1': sum_1 / count}
            max_valor = max(average_dict.values())
            maximo = [clave for clave, valor in average_dict.items() if valor == max_valor]
            if(value in maximo):
                aux += 1
        
        if aux == len(selected_condition):
            nose = 2
        elif aux == 0:
            nose = 0
        else:
            nose = 1

        if nose >= actual:
            actual = nose
            result, rdict = expand(selected_dicts, keys=selected_dicts[0].keys())
            column_index = {format(i, '0' + str(len(selected_dicts)) + 'b')[::-1]: 0 for i in range(2 ** len(selected_dicts))}
            for key in rdict:
                for key2 in column_index:
                    column_index[key2] += rdict[key][key2] / len(rdict.keys())
            rdict = {cadenas.replace('0', ''): column_index}
        else:
            cache[(distribution_str, key)] = (actual, None, None)
            return actual, None, None

    cache[(distribution_str, key)] = (actual, result, rdict)
    return actual, result, rdict

def calcular_resultado(dict1, dict2, keys, key, cache, row_map, col_map):
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

    all_letters = ''.join(row_map.keys()) + ' = ' + key

    if not new_keys1 and new_keys2:
        for key in keys:
            selected_keys2 = [key[col_map[let]] for let in new_keys2]
            key2 = ''.join(selected_keys2)
            value2 = next(iter(dict2.values()))[key2]
            resultado[key] = value2
        resultadoF[all_letters] = resultado
        cache[cache_key] = (resultadoF, all_letters)
        return resultadoF, all_letters
    elif not new_keys2 and new_keys1:
        for key in keys:
            selected_keys1 = [key[col_map[let]] for let in new_keys1]
            key1 = ''.join(selected_keys1)
            value1 = next(iter(dict1.values()))[key1]
            resultado[key] = value1
        resultadoF[all_letters] = resultado
        cache[cache_key] = (resultadoF, all_letters)
        return resultadoF, all_letters

    for key in keys:
        selected_keys1 = [key[col_map[let]] for let in new_keys1 if let != '∅']
        selected_keys2 = [key[col_map[let]] for let in new_keys2 if let != '∅']
        key1 = ''.join(selected_keys1)
        key2 = ''.join(selected_keys2)

        value1 = next(iter(dict1.values())).get(key1, 1)
        value2 = next(iter(dict2.values())).get(key2, 1)
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

def procesar_diccionario(dic):
    nuevo_dic = {}
    for clave, valor in dic.items():
        # Contar la frecuencia de cada carácter en el valor
        contador = {}
        for char in valor:
            if char in contador:
                contador[char] += 1
            else:
                contador[char] = 1
        
        # Encontrar el carácter con la frecuencia máxima
        max_frecuencia = max(contador.values())
        max_caracteres = [k for k, v in contador.items() if v == max_frecuencia]
        
        # Si hay solo un carácter con la máxima frecuencia, actualizar el valor
        if len(max_caracteres) == 1:
            nuevo_dic[clave] = max_caracteres[0]
    
    return nuevo_dic

def invertir_diccionario(dic):
    invertido = {}
    for clave, valor in dic.items():
        if valor in invertido:
            raise ValueError("Los valores no son únicos, no se puede invertir el diccionario.")
        invertido[valor] = clave
    return invertido

def heuristic(dicts, key, letters, threshold=0.48):
    letters = letters.replace('0', '')
    combinaciones, combinacionesOP = obtener_cadena_valores(letters)

    row_letters = ''.join([c for c in letters.split('/')[1] if c != '0'])
    col_letters = ''.join([c for c in letters.split('/')[0] if c != '0'])

    row_map = {char: idx for idx, char in enumerate(row_letters)}
    col_map = {char: idx for idx, char in enumerate(col_letters)}
    empty_col = ''.join(['0' for _ in col_map])
    empty_row = ''.join(['0' for _ in row_map])

    menor = np.inf
    best_partition = ""
    best_dict = None
    actual1 = 0      # 0: bad, 1: regular, 2: good
    actual2 = 0      # 0: bad, 1: regular, 2: good
    conditions = {}
    suma = 0

    num_combinaciones = 2 ** len(col_map)
    column_index = {
        format(i, '0' + str(len(col_map)) + 'b')[::-1]: idx
        for idx, i in enumerate(range(num_combinaciones))
    }

    keys = column_index
    n = len(keys)
    distance_matrix = np.zeros((n, n))

    ordic2 = {}
    original, ordic = expand(dicts, keys=dicts[0].keys(), key=key)
    
    dict_key = letters.split('/')[1] + " = " + key
    ordic2[dict_key] = ordic

    pdox = pd.DataFrame.from_dict(ordic2).transpose()
    print("\nOriginal:\n\n", pdox) 

    # ordic['00001'] = 0.25
    # ordic['00011'] = 0.25
    # ordic['00101'] = 0.25
    # ordic['00111'] = 0.25
    ordenado = dict(sorted(ordic.items(), key=lambda item: item[1], reverse=True))
    ordenado = list(ordenado.items())

    i = 0
    while suma < threshold:
        clave, valor = ordenado[i]
        if valor == 0:
            break
        suma += valor
        i += 1
        for j, let in enumerate(clave):
            if j in conditions:
                conditions[j] += let
            else:
                conditions[j] = let

    conditions = procesar_diccionario(conditions)
    inverted = invertir_diccionario(col_map)
    nuevo_diccionario = {inverted[key]: value for key, value in conditions.items()}

    for i, keyy in enumerate(keys):
        for j, key2 in enumerate(keys):
            distance_matrix[i, j] = hamming_distance(keyy, key2)

    cache = {}

    for combinacion, combinacion_op in zip(combinaciones, combinacionesOP):
        actual1, result1, d1 = getDistribution(combinacion, dicts, key=key, cache=cache, row_map=row_map, col_map=col_map, conditions=nuevo_diccionario, actual=actual1)
        actual2, result2, d2 = getDistribution(combinacion_op, dicts, key=key, cache=cache, row_map=row_map, col_map=col_map, conditions=nuevo_diccionario, actual=actual2)
        if result1 is not None and result2 is not None:
            rx, letters = calcular_resultado(d1, d2, keys.keys(), key=key, cache=cache, row_map=row_map, col_map=col_map)
            pdrx = pd.DataFrame.from_dict(rx).transpose()
            values1 = pdrx.iloc[0].values
            emd_distance = calculate_emd(values1, original, distance_matrix)

            if emd_distance < menor:
                menor = emd_distance

                combinacion = combinacion.split('/')
                combinacion[0] = combinacion[0].replace(empty_col, '∅')
                combinacion[1] = combinacion[1].replace(empty_row, '∅')
                combinacion = '/'.join(combinacion)
                combinacion_op = combinacion_op.split('/')
                combinacion_op[0] = combinacion_op[0].replace(empty_col, '∅')
                combinacion_op[1] = combinacion_op[1].replace(empty_row, '∅')
                combinacion_op = '/'.join(combinacion_op)

                best_partition = combinacion + " * " + combinacion_op
                best_dict = pdrx
                if menor == 0:
                    break
    
    best_partition = best_partition.replace('0', '')

    print("\n\nMenor EMD: ", menor)
    print("Mejor partición: ", best_partition)
    print("Fila de la partición:\n\n", best_dict)
    return pdox, menor, best_partition, best_dict