#Denne python fil skal ekstrakte de features jeg skal bruge til at træne min model, ud fra den rå data
#Måden jeg vil ekstrakte dataen på er at segmenterer den rå data op i segmenter af 200ms, hvilket svarer til 40 datapunkter, hvor hver af de 8 (eller 4) kanaler (fra Myoarmbåndet)

#Import af libraries
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import ast

#Sti til data og omdannelse til pandas Dataframe
S2_C1_2_Tr1 = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None)


segment_list = []
sleep = time.sleep
np_window_datapoints = []

def get_ssc(np_window_datapoints):
    np_window_datapoints = np.array(np_window_datapoints)
    n_ssc = 0

    datapoints_difference = np.diff(np_window_datapoints)
    indices_to_delete = []
    for i, elem in enumerate(datapoints_difference):
        
        if elem == 0 :
            indices_to_delete.append(i)

    datapoints_difference_no_zero = np.delete(datapoints_difference, indices_to_delete)
    

    for i, elem in enumerate(np.diff(np.sign(datapoints_difference_no_zero))):
        #print(np.diff(np.sign(datapoints_difference_no_zero)))
        if np.abs(np.diff(np.sign(datapoints_difference_no_zero)))[i] == 2:
            n_ssc+=1
    return n_ssc

def get_zero_crossings(np_window_datapoints, ZC_position=False):
    np_window_datapoints = np.array(np_window_datapoints)
    #Datapoints fra det vinduet, hvor alle nuller er sorteret fra gemmes i dette array
    window_datapoints_no_zero = []
    #Dette array indeholder positionen hvor alle zero-crossings er i window_datapoints
    window_datapoints_ZC_positions = []
    #Dette array bruges til at sammenligne 2 tal
    comparer_arr = []
    #Denne variabel kommer til at indeholde mængden af zero-crossings
    n_of_ZC = 0

    #Sortér alle nuller fra window_datapoints således man nemmere kan beregne mængden af zc's
    for elem in np_window_datapoints:

        if elem != 0:
            window_datapoints_no_zero = np.append(window_datapoints_no_zero, elem)

    window_datapoints_no_zero_difference_signs = np.diff(np.sign(window_datapoints_no_zero))
    
    for elem in window_datapoints_no_zero_difference_signs:
        if np.abs(elem) == 2:
            n_of_ZC+=1
            
    

    
    i=0
    for cell in np.diff(np.sign(np_window_datapoints)):
            
        if np.abs(cell) == 2:
            window_datapoints_ZC_positions = np.append(window_datapoints_ZC_positions, i)
                
        elif np.abs(cell) == 1:
            comparer_arr.append(cell)

            if len(comparer_arr) == 2:

                if comparer_arr[0] == comparer_arr[1]:
                    window_datapoints_ZC_positions = np.append(window_datapoints_ZC_positions, i)
                        
                comparer_arr.clear()
        i+=1
        #print(window_datapoints_ZC_positions)

    if ZC_position == True:
        return [n_of_ZC, window_datapoints_ZC_positions]
    else :
        return n_of_ZC
        
  

def feature_extractor(np_window_datapoints):

    #Mean absolute value
    #MAV er bekræftet til at være sand 24/10
    MAV = np.mean(np.abs(np_window_datapoints))

    #Zero crossings. Hvor signalet krydser 0mV
    ZC  = get_zero_crossings(np_window_datapoints)

    WL =  np.sum(np.abs(np.diff(np_window_datapoints)))

    SSC = get_ssc(np_window_datapoints)

    feature_list = [MAV, ZC, WL, SSC]
    return feature_list



#Læs dataframen
read = pd.read_csv('array.csv')
print(type(ast.literal_eval(read.iloc[0, 0])))

# Brug ast.literal_eval til at konvertere stringen tilbage til en liste
#Denne variabel bliver kun brugt til at determinere længden af en celle i loopet herunder
df_window_list = ast.literal_eval(read.iloc[0, 0])


for j in range(len(read.columns)): #Looper over kolonnerne i read
    
    for i in range(len(df_window_list)): #Looper over rækkerne
        sleep(0.3)
        
        print(feature_extractor(ast.literal_eval(read.iloc[i, j])))
        # for punkt, datapoint in enumerate(ast.literal_eval(read.iloc[i, j])):
        #     print(f"punkt{punkt} {datapoint}")
        #     feature_extractor()
        # #print(df_window_list)
        print(f"Række nr: {i}")
        
    print(f"Row: {i}, Column: {j}")
    sleep(0.5)
    


# # Convert the list to a NumPy array
# r_array = np.array(r_list)

# # Now r_array is a NumPy array, and you can iterate through it as you would normally
# for elem in r_array:
#     print(elem)