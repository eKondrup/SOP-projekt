import pandas as pd
import numpy as np
import ast

list = [[-0.0078125, 0.0078125, 0.       , 0.       , 0.       , 0.0078125,
 -0.0078125,-0.0078125,-0.0078125,-0.015625 ,-0.0078125, 0.0078125,
 -0.015625 ,-0.015625 , 0.       ,-0.0078125,-0.0078125, 0.       ,
 -0.0078125, 0.       , 0.       ,-0.0078125,-0.0078125,-0.015625 ,
 -0.0078125, 0.0078125, 0.       , 0.0078125, 0.       ,-0.015625 ,
  0.       ,-0.0078125, 0.       ,-0.0078125, 0.       ,-0.015625 ,
  0.0078125, 0.       ,-0.0078125, 0.       ]]
print(list)
df = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None)



# Read the DataFrame
read = pd.read_csv('array.csv')

# Get the first cell from the first column
r = read.iloc[0, 0]

# Use ast.literal_eval to convert the string back to a list
r_list = ast.literal_eval(r)

# Convert the list to a NumPy array
r_array = np.array(r_list)

# Now r_array is a NumPy array, and you can iterate through it as you would normally
for elem in r_array:
    print(elem)

#print(r)
#r = pd.DataFrame(r)
#print(r)
#print(pd.DataFrame(r).to_numpy())

'''
window_size = 40
overlap = 10
step_size = window_size - overlap

# Initialize an empty list to store segments
segments = []

# Loop through each column
for col in df.columns:
    col_data = df[col].values
    col_segments = []
    
    # Create segments for the current column
    for start in range(0, len(col_data) - window_size + 1, step_size):
        end = start + window_size
        segment = col_data[start:end]
        col_segments.append(segment)
        
    segments.append(col_segments)

# Convert the list of segments to a 3D NumPy array
array_3d = np.array(segments)
array_3d = np.transpose(array_3d, (1, 0, 2))

# Initialize an empty DataFrame
new_df = pd.DataFrame()

# Populate the DataFrame with segments
for i in range(array_3d.shape[1]):  # Loop through each column
    col_data = []
    for j in range(array_3d.shape[0]):  # Loop through each segment
        segment = array_3d[j, i, :]
        # Convert the segment to a string, setting max_line_width to a large value
        segment_str = np.array2string(segment, separator=',', max_line_width=np.inf)
        col_data.append(segment_str)
    new_df[f'Column_{i+1}'] = col_data

# Write the DataFrame to a new CSV file
new_df.to_csv('array.csv', index=False)
'''

'''
#data = pd.DataFrame(data).to_numpy
#print(np.count_nonzero(np.diff(npData.iloc[:,0])))
comp = np.array([])
n_ssc = i=0

#data = np.array([1,2,4,8,16, 12, 13.12341234, -2,-2,-3,-2,0,0,0,2,1])
data = np.array([["1,1,1","1,1,1","1,1,1"],["2,2,2","2,2,2","2,2,2", "2,2,2"], ["3,3,3","3,3,3","3,3,3"], ["1,1,1","1,1,1","1,1,1"],["2,2,2","2,2,2","2,2,2"], ["3,3,3","3,3,3","3,3,3"]])
read = pd.read_csv("array.csv", header=None)
read = pd.DataFrame(read)
print(read)
df = pd.DataFrame(data)
print(df)

df.to_csv('array.csv', mode='a', index=False, header=False)



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