import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

data = pd.read_csv('../EMG data/S2_CSV/S2_G1_40_Tr1.csv', header=None)

data = data.iloc[:,0]
# data = np.abs(np.diff(data))    

gesture = "G1"
trail = "1"
df = pd.read_csv(f'../EMG data/{gesture}/S2_{gesture}_40_Tr{trail}.csv', header=None)
#Skal kun bruge datapunkter fra de 5 sekunder i midten af datasættet
df = df.iloc[1000:2000,0]

# plt.figure(figsize=(14, 6))
# plt.xlim(0, 40)  # x-axis will go from 0 to 10
# plt.ylim(-0.8, 0.8)  # y-axis will go from 0 to 10
# plt.plot(df)
# plt.plot(data)
# plt.title('EMG Signal')
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

plt.figure(figsize=(14, 6))
# plt.xticks([0, 5, 10, 15, 20, 25,30,35,40])
plt.yticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
# plt.xlim(0, 40)  # x-axis will go from 0 to 10
plt.ylim(-0.8, 0.8)  # y-axis will go from 0 to 10
plt.plot(data)
plt.plot(df)
plt.title('EMG Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()