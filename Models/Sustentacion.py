import numpy as np
import pandas as pd
import Matrix_PD_version as mpd

def subsystem(dict, subsystem):
    column_indices = [i for i, c in enumerate(subsystem.split('/')[0]) if c == '0']
    row_indices = [i for i, c in enumerate(subsystem.split('/')[1]) if c == '0']

    for index in sorted(column_indices, reverse=True):
        dict = mpd.marginalize_column(dict, index)
    
    new_dict = dict
    keys = list(dict.keys())
    
    for index in sorted(row_indices, reverse=True):
        aux_dict = {}
        for key in keys:
            if key[index] == '0':
                aux_dict[key[:index] + key[index + 1:]] = new_dict[key]
        keys = list(aux_dict.keys())
        new_dict = aux_dict

    PD_dict = pd.DataFrame(new_dict).transpose()
    print("Subsistema\n")
    print(PD_dict)

    column_indices = sum(1 for caracter in subsystem.split('/')[0] if caracter != '0')

    dicts = getIndividualMatrixes(new_dict, column_indices)

    print("\nIndividuales\n")
    for dict in dicts:
        pd_dict = pd.DataFrame(dict).transpose()
        print(pd_dict)
        print("\n")

    return new_dict, dicts

def getIndividualMatrixes(dict, column_indices):
    dicts = []
    for index in range(column_indices):
        aux_dict = dict
        for index2 in range(column_indices, 0, -1):
            if (index2 - 1) != index:
                aux_dict = mpd.marginalize_column(aux_dict, index2 - 1)
        dicts.append(aux_dict)
    return dicts

ABCD = {
    '0000': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1000': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '0100': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1100': {'0000': 0, '1000': 0, '0100': 0, '1100': 0, '0010': 0, '1010': 0, '0110': 0, '1110': 0, '0001': 0, '1001': 0, '0101': 0, '1101': 0, '0011': 0.49, '1011': 0.21, '0111': 0.21, '1111': 0.09},
    '0010': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1010': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '0110': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1110': {'0000': 0, '1000': 0, '0100': 0, '1100': 0, '0010': 0, '1010': 0, '0110': 0, '1110': 0, '0001': 0, '1001': 0, '0101': 0, '1101': 0, '0011': 0.49, '1011': 0.21, '0111': 0.21, '1111': 0.09},
    '0001': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1001': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '0101': {'0000': 0.2401, '1000': 0.1029, '0100': 0.1029, '1100': 0.0441, '0010': 0.1029, '1010': 0.0441, '0110': 0.0441, '1110': 0.0189, '0001': 0.1029, '1001': 0.0441, '0101': 0.0441, '1101': 0.0189, '0011': 0.0441, '1011': 0.0189, '0111': 0.0189, '1111': 0.0081},
    '1101': {'0000': 0, '1000': 0, '0100': 0, '1100': 0, '0010': 0, '1010': 0, '0110': 0, '1110': 0, '0001': 0, '1001': 0, '0101': 0, '1101': 0, '0011': 0.49, '1011': 0.21, '0111': 0.21, '1111': 0.09},
    '0011': {'0000': 0, '1000': 0, '0100': 0, '1100': 0.49, '0010': 0, '1010': 0, '0110': 0, '1110': 0.21, '0001': 0, '1001': 0, '0101': 0, '1101': 0.21, '0011': 0, '1011': 0, '0111': 0, '1111': 0.09},
    '1011': {'0000': 0, '1000': 0, '0100': 0, '1100': 0.49, '0010': 0, '1010': 0, '0110': 0, '1110': 0.21, '0001': 0, '1001': 0, '0101': 0, '1101': 0.21, '0011': 0, '1011': 0, '0111': 0, '1111': 0.09},
    '0111': {'0000': 0, '1000': 0, '0100': 0, '1100': 0.49, '0010': 0, '1010': 0, '0110': 0, '1110': 0.21, '0001': 0, '1001': 0, '0101': 0, '1101': 0.21, '0011': 0, '1011': 0, '0111': 0, '1111': 0.09},
    '1111': {'0000': 0, '1000': 0, '0100': 0, '1100': 0, '0010': 0, '1010': 0, '0110': 0, '1110': 0, '0001': 0, '1001': 0, '0101': 0, '1101': 0, '0011': 0, '1011': 0, '0111': 0, '1111': 1},
}

PD_ABCD = pd.DataFrame(ABCD).transpose()
print("Matriz completa\n")
print(PD_ABCD)
print("\n\n\n")

# key completa = 1000
new_dict, dicts = subsystem(ABCD, 'ABC0/ABC0')
mpd.bottom_up(dicts[0], dicts[1], dicts[2], key = '100')

# new_dict, dicts = subsystem(ABCD, 'A0C0/AB00')
# mpd.bottom_up(dicts[0], dicts[1], key = '10')

# new_dict, dicts = subsystem(ABCD, 'A0C0/ABC0')
# mpd.bottom_up(dicts[0], dicts[1], key = '100')