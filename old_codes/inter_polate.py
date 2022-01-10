import pandas as pd
import numpy as np
import os

excel_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2020长三角mp0.05_len176.h5'
save_path = excel_path.split('.h5')[0]+'linearinter.h5'
store = pd.HDFStore(excel_path)
keys = store.keys()

for key in keys:
    df = store.get(key)
    
    po_df = df.interpolate()
    po_df.fillna(0, inplace=True)
    # print(f'key:{key}')
    # print(f'df\n{df}')
    # print(f'po_df\n:{po_df}')
    if os.path.exists(save_path):
        print(f'file exist. key:{key}')
        df.to_hdf(save_path, key=key, mode='r+', format='table')
    else:
        print(f'file not exist. key:{key}, save_path:{save_path}')
        df.to_hdf(save_path, key=key, mode='w', format='table')

