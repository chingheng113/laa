import pandas as pd
import numpy as np
import os

data = pd.read_csv('LAA_computation_lcms1_cli_class_s.csv')
data = data.replace({'.':np.nan, '#N/A':np.nan})

print(data.shape)
data = data.dropna(axis=0)
print(data.shape)
data.to_csv('see.csv', index=False)
print('done')