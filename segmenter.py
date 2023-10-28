####################################################
#Denne fil segmenterer dataen og skriver det til en fil
####################################################

import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

sleep = time.sleep

#Sti til data og omdannelse til pandas Dataframe
gesture = "G1"
trail = "1"
file = f'../EMG data/{gesture}/S2_{gesture}_40_Tr{trail}.csv'
df = pd.read_csv(file, header=None)
#Skal kun bruge datapunkter fra de 5 sekunder i midten af datasættet
df = df.iloc[1000:2000]

window_size = 40
overlap = 10
step_size = window_size - overlap

#Liste til at lagre vinduerne
segments = []
####
#Læsning af fil og konvertering til segments
####
#Iterering gennem hver kolonne
for col in df.columns:
    col_data = df[col].values
    col_segments = []
    
    #Lav segmenter for den nuværende kolonne. range() tager 3 argumenter: range(start, stop, step)
    #Grunden til at stop argumentet er  = len(col_data) - window_size + 1, er fordi det er det sidste element i listen
    for start in range(0, len(col_data) - window_size + 1, step_size):
        #Eksempel: når start er 0, så bliver end = 40, hvorefter segmentet bliver fra start (som er 0) til end ( som er 40)
        end = start + window_size
        segment = col_data[start:end]
        #col_segments er et array af et numpy-array med 40 elementer,dvs col_segments[0] er lig med et numpy array
        #Det er fordi segment er en liste, som bliver appendet til en anden liste
        col_segments.append(segment)
        
    segments.append(col_segments)

#Konvertering af listen af segmenter til 3D np array
#3D arrayet har dette format array_3D[0] = datapunkt og array_3D[1] = datapunkt 2 men hvis man laver dette om til en Dataframe bliver dataen
#Plsceret på den forkerte akse, derfor bruger jeg np.transpose, så dataen korrekt kan laves om til en Dataframe
array_3d = np.array(segments)
array_3d = np.transpose(array_3d, (1, 0, 2))

new_df = pd.DataFrame()

# Tilføj segmenter til dataframen
for i in range(array_3d.shape[1]):  #Looper igennem hver colonne
    col_data = []
    for j in range(array_3d.shape[0]):  # Looper igennem hvert segment
        segment = array_3d[j, i, :]
        # Konvertering af segment til en string, da pandas ikke kan lide nested lists
        segment_str = np.array2string(segment, separator=',', max_line_width=np.inf)
        #col_data[i] = '[elem1, elem2]' altså col_data er liste af strings, som faktisk er andre lister
        col_data.append(segment_str)
    new_df[f'Channel_{i+1}'] = col_data

#Dataframen skrives til en ny fil
# print(new_df)
new_df.to_csv(f'../EMG data/{gesture}/{gesture}_segments_Tr{trail}.csv', index=False)