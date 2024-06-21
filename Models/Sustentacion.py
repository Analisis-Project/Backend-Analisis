import numpy as np
import pandas as pd
import Matrix_PD_version as mpd
import Systems as systems

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

# ABCD = systems.fourNodes()
# ABCDEF = systems.sixNodes()
# ABCDEFGH = systems.eigthNodes1()
# ABCDEFGH = systems.eigthNodes2()
ABCDEFGHIJ = systems.tenNodes()

print("\nMatriz completa\n")
# PD_ABCD = pd.DataFrame(ABCD).transpose()
# print(PD_ABCD)

# PD_ABCDEF = pd.DataFrame(ABCDEF).transpose()
# print(PD_ABCDEF)

# PD_ABCDEFGH = pd.DataFrame(ABCDEFGH).transpose()
# print(PD_ABCDEFGH)

PD_ABCDEFGHIJ = pd.DataFrame(ABCDEFGHIJ).transpose()
print(PD_ABCDEFGHIJ)
print("\n\n")

# new_dict, dicts = subsystem(ABCD, 'ABC0/ABC0')
# mpd.bottom_up(dicts, key = '100', letters='ABC0/ABC0')                      

# new_dict, dicts = subsystem(ABCDEF, '0BCDEF/AB00EF')
# mpd.bottom_up(dicts, key = '1000', letters='0BCDEF/AB00EF')     

# new_dict, dicts = subsystem(ABCDEFGH, 'ABCDEFGH/ABCDEFGH')
# mpd.bottom_up(dicts, key = '10000000', letters='ABCDEFGH/ABCDEFGH')       

new_dict, dicts = subsystem(ABCDEFGHIJ, 'A0C0E0G0I0/0B0D0F0HIJ')
mpd.bottom_up(dicts, key = '100000', letters='A0C0E0G0I0/0B0D0F0HIJ')         