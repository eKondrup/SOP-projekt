#Denne python fil skal ekstrakte de features jeg skal bruge til at træne min model, ud fra den windows lavet af segmenter.py
#Måden jeg vil ekstrakte dataen på er at segmenterer den rå data op i segmenter af 200ms, hvilket svarer til 40 datapunkter, hvor hver af de 8 (eller 4) kanaler (fra Myoarmbåndet)

#Import af libraries
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import ast
import csv

#Sti til data og omdannelse til pandas Dataframe
gesture = "G1"
trail = "1"
file = f'../EMG data/{gesture}/{gesture}_segments_Tr{trail}.csv'
df_read_file = pd.read_csv(file)



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

    #Waveform length
    WL =  np.sum(np.abs(np.diff(np_window_datapoints)))

    #Slope sign changes
    SSC = get_ssc(np_window_datapoints)

    feature_list = [MAV, ZC, WL, SSC]
    return feature_list


################
#Læsning og feature extraction på segments. 
#Features bliver gemt i feature_list, som har en shape på 8*33*4
################

#Denne variabel bliver kun brugt til at determinere længden af en celle i loopet herunder
df_window_list_len = len(df_read_file.iloc[:,0])
feature_list = []

for j in range(len(df_read_file.columns)): #Looper over kolonnerne i read

    col_feature = []
    
    for i in range(df_window_list_len): #Looper over rækkerne
    
    
        # Brug ast.literal_eval til at konvertere stringen tilbage til en liste
        feature = feature_extractor(ast.literal_eval(df_read_file.iloc[i, j]))
        
        print(f"Række nr: {i}")
        
        col_feature.append(feature)     
    
    feature_list.append(col_feature)
    
       
        
    print(f"Kolonne: {j}")

#Konvertering af listen af features til 3D np array
#3D arrayet har dette format array_3D[0] = features1 og array_3D[1] = features2 men hvis man laver dette om til en Dataframe bliver dataen
#Plsceret på den forkerte akse, derfor bruger jeg np.transpose, så dataen korrekt kan laves om til en Dataframe
feature_list_3D = np.array(feature_list)
feature_list_3D = np.transpose(feature_list_3D, (1, 0, 2))

df_features = pd.DataFrame()

# Tilføj features til dataframen. der skal skrives til csv filen
for i in range(feature_list_3D.shape[1]):  #Looper igennem hver colonne
    
    np_col_features = []
    
    for j in range(feature_list_3D.shape[0]):  # Looper igennem hvert segment
        
        np_feature_list = feature_list_3D[j, i, :]
        
        # Konvertering af features til en string, da pandas ikke kan lide nested lists
        np_feature_list_str = np.array2string(np_feature_list, separator=',', max_line_width=np.inf)
        
        #col_data[i] = '[elem1, elem2]' altså np_col_features er liste af strings, som faktisk er andre lister
        np_col_features.append(np_feature_list_str)
    
    #np_col_features skrives til dataframen
    df_features[f'Channel_{i+1}'] = np_col_features

#features skrives til csv
df_features.to_csv(f'../EMG data/{gesture}/{gesture}_features_Tr{trail}.csv', index=False)