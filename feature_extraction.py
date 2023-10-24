#Denne python fil skal ekstrakte de features jeg skal bruge til at træne min model, ud fra den rå data
#Måden jeg vil ekstrakte dataen på er at segmenterer den rå data op i segmenter af 200ms, hvilket svarer til 40 datapunkter, hvor hver af de 8 (eller 4) kanaler (fra Myoarmbåndet)

#Import af libraries
import pandas as pd
import numpy as np
import time 


#Sti til data
S2_C1_2_Tr1 = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None)

#Adskillelse af channels/kolonner
col1_S2_C1_2_Tr1 = S2_C1_2_Tr1.iloc[:,0]

#Konvertering til Pandas DataFrame
df = pd.DataFrame(S2_C1_2_Tr1)

segment_list = []
sleep = time.sleep
#MAV er bekræftet til at være sand 24/10
def feature_extractor(window_datapoints):
    MAV = np.mean(np.abs(window_datapoints))
    ZC  = np.sum(np.diff(np.sign(window_datapoints)))
    WL =  np.sum(np.abs(np.diff(window_datapoints)))
    SSC = diff_signal = np.diff(window_datapoints)
    feature_list = [MAV, ZC, WL, SSC]
    return feature_list

#Iterer over kolonner i dataframen
i=0
for column in df.columns:
    segment_list.clear()
    print(f"Kolonne: {column+1} {segment_list}")
    sleep(0.1)
    #Iterer over celler i hver kolonne
    for cell in df.iloc[:,column]:
       
        if len(segment_list) < 40:
            segment_list.append(cell)
        else:
            window_datapoints = pd.DataFrame(segment_list).to_numpy().flatten()
            print(window_datapoints)
            i+=1
            print(f"Segment nr: {i} Segment længde: {len(segment_list)} MAV: {feature_extractor(window_datapoints)[0]} ZC: {feature_extractor(window_datapoints)[1]} WL: {feature_extractor(window_datapoints)[2]} SSC: {feature_extractor(window_datapoints)[3]}")
            
            #segment_list.clear()
            print(np.sign(window_datapoints))
            print(f"diff: {np.diff(np.sign(window_datapoints))}")
            segment_list = segment_list[-10:]
            sleep(1)