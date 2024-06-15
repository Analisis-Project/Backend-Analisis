import numpy as np
import pandas as pd

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

def expand(*dicts, keys, key=None):
    n = len(keys)
    key_to_index = {k: i for i, k in enumerate(keys)}
    column_index = {k: i for i, k in enumerate(dicts[0].keys())}
    
    if key:
        if key not in column_index:
            raise ValueError(f"Key '{key}' not found in dictionaries")
        
        row = np.zeros(n, dtype=np.float16)
        for key2 in keys:
            value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
            row[key_to_index[key2]] = value
        return row
    else:
        state_matrix = np.zeros((len(dicts[0]), n), dtype=np.float16)
        for idx, key in enumerate(dicts[0]):
            for key2 in keys:
                value = np.prod([d[key][c] for d, c in zip(dicts, key2)])
                state_matrix[idx, key_to_index[key2]] = value
        return state_matrix

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

# # EJEMPLO DE USO DE LAS FUNCIONES
# expanded_matrix = expand(Af, Bf, Cf, keys=Af.keys())
# print("Matriz expandida")
# print(expanded_matrix)

# result_df = pd.DataFrame(expanded_matrix, index=Af.keys(), columns=Af.keys())
# print("\nDataframe de la matriz expandida")
# print(result_df)

# # Ejemplo de uso con una clave específica
# key = '010'
# specific_row = expand(Af, Bf, Cf, keys=Af.keys(), key=key)
# print("\nSacar fila especifica sin tener que calcular toda la matriz, Fila: ", key)
# print(specific_row)

# specific_row_df = pd.DataFrame(specific_row, index=Af.keys(), columns=[key]).transpose()
# print("\nDataframe de la fila especifica")
# print(specific_row_df)

# transformed_dicts = marginalize_row([Af, Bf, Cf], index=0)

# print("\nDatos de entrada después de marginalizar A")
# print("A: ", transformed_dicts[0])
# print("B: ", transformed_dicts[1])
# print("C: ", transformed_dicts[2])

# print("\nMatriz expandida después de marginalizar")
# expanded_matrix = expand(transformed_dicts[0], transformed_dicts[1], transformed_dicts[2], keys=Af.keys())
# print(expanded_matrix)

# print("\nDataframe de la matriz expandida después de marginalizar")
# result_df = pd.DataFrame(expanded_matrix, index=transformed_dicts[0].keys(), columns=Af.keys())
# print(result_df)


#EJEMPLO DEL TALLER EN CLASE BACKTRACKING PUNTO 2
print("B original")
print(Bf)

print("\nEn B marginalizar A")
transformed_B = marginalize_row([Bf], index=0)
print(transformed_B[0])

keyys = transformed_B[0].keys()

print("\nEn B marginalizar C")
transformed_B = marginalize_row([transformed_B][0], index=0)
print(transformed_B[0])

print("\nC original")
print(Cf)

print("\nEn C marginalizar A")
transformed_C = marginalize_row([Cf], index=0)
print(transformed_C[0])

print("\nEn C marginalizar C")
transformed_C = marginalize_row([transformed_C[0]], index=0)
print(transformed_C[0])

key = '0'

print("\nRESULTADOS EN FORMA DE MATRIZ")
print("Producto tensor")
result1 = expand(transformed_B[0], transformed_C[0], keys = keyys)
print(result1)

print("\nProducto tensor cuando C vale 0")
result2 = expand(transformed_B[0], transformed_C[0], keys = keyys, key = key)
print(result2)

print("\nRESULTADOS EN FORMA DE DATAFRAME")
print("Producto tensor")
result_df = pd.DataFrame(result1, index=transformed_C[0].keys(), columns=keyys)
print(result_df)

print("\nProducto tensor cuando C vale 0")
result_df = pd.DataFrame(result2, index=keyys, columns=[key]).transpose()
print(result_df)
