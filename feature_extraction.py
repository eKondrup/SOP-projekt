#Denne python fil skal ekstrakte de features jeg skal bruge til at træne min model, ud fra den rå data
#Måden jeg vil ekstrakte dataen på er at segmenterer den rå data op i segmenter af 200ms, hvilket svarer til 40 datapunkter, hvor hver af de 8 (eller 4) kanaler (fra Myoarmbåndet)

#Import af libraries
import pandas as pd
import numpy as np


#Data lokation
S2_C1_2_Tr1 = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv')

#Adskillelse af channels/kolonner
col1_S2_C1_2_Tr1 = S2_C1_2_Tr1.iloc[:,0]



print(S2_C1_2_Tr1.head)

