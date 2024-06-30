import numpy as np
import pandas as pd
import Matrix_PD_version as mpd
import Matrix_Heuristic as mhd
import Systems as systems
import time

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


# ABC = systems.threeNodes()
# ABCD = systems.fourNodes()
# ABCDE = systems.fiveNodes()
# ABCDEF = systems.sixNodes()
# ABCDEFGH = systems.eigthNodes1()
ABCDEFGH = systems.eigthNodes2()
# ABCDEFGHIJ = systems.tenNodes()

print("\nMatriz completa\n")
# PD_ABC = pd.DataFrame(ABC).transpose()
# print(PD_ABC)

# PD_ABCD = pd.DataFrame(ABCD).transpose()
# print(PD_ABCD)

# PD_ABCDEF = pd.DataFrame(ABCDEF).transpose()
# print(PD_ABCDEF)

PD_ABCDEFGH = pd.DataFrame(ABCDEFGH).transpose()
print(PD_ABCDEFGH)

# PD_ABCDEFGHIJ = pd.DataFrame(ABCDEFGHIJ).transpose()
# print(PD_ABCDEFGHIJ)
print("\n\n")

# mpd.bottom_up(ABC, key = '100', letters='ABC/ABC')

# new_dict, dicts = subsystem(ABCD, 'ABC0/ABC0')
# mpd.bottom_up(dicts, key = '100', letters='ABC0/ABC0')        
 
# mpd.bottom_up(ABCDE, key = '10001', letters='ABCDE/ABCDE')              

# new_dict, dicts = subsystem(ABCDEF, 'ABCDEF/ABCDEF')

# inicioH = time.time()
# mhd.heuristic(dicts, key = '100000', letters='ABCDEF/ABCDEF')
# finH = time.time()
# inicioPD = time.time()
# mpd.bottom_up(dicts, key = '100000', letters='ABCDEF/ABCDEF')     
# finPD = time.time()

# print("Tiempo Heuristica: ", finH - inicioH)
# print("Tiempo Programacion Dinamica: ", finPD - inicioPD)

new_dict, dicts = subsystem(ABCDEFGH, 'ABCDEFGH/ABCDEFGH')
# inicioPD = time.time()
# mpd.bottom_up(dicts, key = '10000000', letters='ABCDEFGH/ABCDEFGH')   
# finPD = time.time()
# print("Tiempo Programacion Dinamica: ", finPD - inicioPD)
inicioH = time.time()
mhd.heuristic(dicts, key = '10000000', letters='ABCDEFGH/ABCDEFGH')
finH = time.time()
print("Tiempo Heuristica: ", finH - inicioH)

# new_dict, dicts = subsystem(ABCDEFGHIJ, 'A0C0E0G0I0/0B0D0F0HIJ')
# mpd.bottom_up(dicts, key = '100000', letters='A0C0E0G0I0/0B0D0F0HIJ')         