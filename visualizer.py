import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

data = pd.read_csv('../EMG data/S2_CSV/S2_C1_2_40_Tr1.csv', header=None)

plt.figure(figsize=(14, 6))
plt.plot(data.iloc[:,0])
plt.title('EMG Signal')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

