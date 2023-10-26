import numpy as np
import pandas as pd

data = np.array([1,2,4,8,16, 12, 13.12341234, -2,-2,-3,-2,0,0,0,2,1])

npData = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None).iloc[:,0]
#data = pd.DataFrame(data).to_numpy
#print(np.count_nonzero(np.diff(npData.iloc[:,0])))
comp = np.array([])
n_ssc = i=0

data = np.array([1,2,4,8,16, 12, 13.12341234, -2,-2,-3,-2,0,0,0,2,1])
ddata = np.diff(data)
to_del = []
print(np.sign(np.diff(data)))
print(np.diff(np.sign(ddata)))
i = 0
for i, elem in enumerate(ddata):
    print(elem)
    if elem == 0:
        to_del.append(i)
if to_del:
    ddata_z = np.delete(ddata, to_del)
print(ddata_z)

'''
for elem in np.diff(np.sign(ddata)):
    if np.abs(np.diff(np.sign(ddata)))[i] == 2:
        n_ssc+=1
    i+=1
print(n_ssc)

def get_ssc(np_window_datapoints):
    i_ssc = 0
    n_ssc = 0
    for elem in np.diff(np.sign(np.diff(np_window_datapoints))):
        if np.abs(np.diff(np.sign(np.diff(np_window_datapoints))))[i_ssc] == 2:
            n_ssc+=1
        i_ssc+=1
    return n_ssc

print(get_ssc(np.array(npData.iloc[40,0])))'''