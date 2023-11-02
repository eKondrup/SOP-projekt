import pandas as pd
import numpy as np
import ast
import icecream
import time
sleep = time.sleep

list = [-0.0078125, 0.0078125, 0.       , 0.       , 0.       , 0.0078125,
 -0.0078125,-0.0078125,-0.0078125,-0.015625 ,-0.0078125, 0.0078125,
 -0.015625 ,-0.015625 , 0.       ,-0.0078125,-0.0078125, 0.       ,
 -0.0078125, 0.       , 0.       ,-0.0078125,-0.0078125,-0.015625 ,
 -0.0078125, 0.0078125, 0.       , 0.0078125, 0.       ,-0.015625 ,
  0.       ,-0.0078125, 0.       ,-0.0078125, 0.       ,-0.015625 ,
  0.0078125, 0.       ,-0.0078125, 0.       ]
#Sti til data og omdannelse til pandas Dataframe
gesture = "G1"
trail = "1"
test_file = f'../EMG data/{gesture}/{gesture}_features_Tr{trail}.csv'
test_file =f"C:/Users/emila/OneDrive - TECHCOLLEGE/Skole/SOP/EMG data/G1/{gesture}_features_Tr{trail}.csv"
df_read_test_file = pd.read_csv(test_file)

trail = "1"

gesture1 = "G1"
test_file1 = f'../EMG data/{gesture1}/{gesture1}_features_Tr{trail}.csv'
test_file1 = pd.read_csv(test_file1)

gesture2 = "G1"
# trail2 = "1"
test_file2 = f'../EMG data/{gesture2}/{gesture2}_features_Tr{trail}.csv'
test_file2 = pd.read_csv(test_file2)

gesture3 = "G1"
# trail3 = "1"
test_file3 = f'../EMG data/{gesture3}/{gesture3}_features_Tr{trail}.csv'
test_file3 = pd.read_csv(test_file3)

gesture4 = "G1"
# trail4 = "1"
test_file4 = f'../EMG data/{gesture4}/{gesture4}_features_Tr{trail}.csv'
test_file4 = pd.read_csv(test_file4)

gesture5 = "G1"
# trail5 = "1"
test_file5 = f'../EMG data/{gesture5}/{gesture5}_features_Tr{trail}.csv'
test_file5 = pd.read_csv(test_file5)

n_features = 4
n_segments = 33
n_channels = 8

def func(list):
    for elem in list:
        if elem == 0:
            if elem == 0:
                return elem
        else: return

print(func(list=list))




# df_window_list_len = len(df_read_test_file.iloc[:,0])
# feature_list = []

# for j in range(len(df_read_test_file.columns)): #Looper over kolonnerne i read

#     col_feature = []

#     for i in range(df_window_list_len): #Looper over rækkerne
#         # Brug ast.literal_eval til at konvertere stringen tilbage til en liste
#         feature = ast.literal_eval(df_read_test_file.iloc[i, j])
                       
#         col_feature.append(feature)     

#     feature_list.append(col_feature)

# feature_list_3D = np.array(feature_list)

# feature_list_2d = feature_list_3D.reshape(-1, n_features)

# df_read_test_file = df_read_test_file.to_numpy
# i=0
# feature_list = []
# feature_list = np.empty((0,32))

# for feature_list_str in df_read_test_file.values: #Looper over rækken i read
#     feature_list_row = []
#     for elem in feature_list_str:
#         # print(f"type: {type(elem)} elem: {elem}")
#         feature = ast.literal_eval(elem)
#         # print(f"type: {type(feature)} elem: {feature}")
#         feature_list_row = np.append(feature_list_row, feature)

#         # print(f"{i} {feature_list_row}")
#         i+=1
#         # sleep(1)

#     feature_list = np.vstack((feature_list, feature_list_row))
# print(feature_list.shape)
# print(pd.DataFrame(feature_list))
# print(pd.DataFrame(feature_list))
        # for cell in elem:
        #     print(f"{feature}")
        #     feature = ast.literal_eval([cell])
        #     print(feature)
# ar = df_read_test_file.values
# print(f"values {ar}")
# ar = np.transpose(ar)

# print(f"transposed: {ar}")



# # Laver en liste med hver channel i [0,1..,7]
# #Der bliver brugt en list comprehension: https://stackoverflow.com/questions/6475314/python-for-in-loop-preceded-by-a-variable
# channels_list = np.array([l for l in range(n_channels)]).flatten()

# #Listen bliver gentaget n_segmetns n_segments gange og samme elementer
# #Sættes ved siden af hinanden [0*33,1*33...] ish..
# channels_list = np.repeat(channels_list, n_segments)

# # # Add column indices to the 2D array
# channels_list_reshaped = channels_list[:feature_list_2d.shape[0]].reshape(-1, 1)
# print(channels_list_reshaped)

# channels_list_reshaped = channels_list[263].reshape(-1, 1)
# print(channels_list_reshaped)


# d = np.array([[[1 , 2 , 3],[1 , 2 , 3]],[[1 , 2 , 3],[1 , 2 , 3]], [[1, 2, 3],[1, 2, 3]],[[1, 2, 3],[1, 2, 3]]])

# l = d.reshape(-1,3)

# # icecream(print(d))
# print(d)

# print(l)
# # Read the DataFrame
# read = pd.read_csv('array.csv')

# # Get the first cell from the first column
# r = read.iloc[0, 0]

# # Use ast.literal_eval to convert the string back to a list
# r_list = ast.literal_eval(r)

# # Convert the list to a NumPy array
# r_array = np.array(r_list)

# # Now r_array is a NumPy array, and you can iterate through it as you would normally
# for elem in r_array:
#     print(elem)

# #print(r)
# #r = pd.DataFrame(r)
# #print(r)
# #print(pd.DataFrame(r).to_numpy())

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

# Write the DataFrame to a new CSV test_file
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