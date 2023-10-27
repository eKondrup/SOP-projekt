import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

data = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None)

data = data.iloc[:,0]
# data = np.abs(np.diff(data))    

plt.figure(figsize=(14, 6))
plt.xlim(0, 40)  # x-axis will go from 0 to 10
plt.ylim(-0.8, 0.8)  # y-axis will go from 0 to 10
plt.plot(data)
plt.title('EMG Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
