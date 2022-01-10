import pandas as pd
import numpy as np
import os
# import collections

excel_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2018.h5'
save_path = excel_path.split('.h5')[0]+'长三角mp0.05_len176.h5'
print(save_path)
f = open('./长三角站点编码mp0.05_len176.txt', 'r')
con = f.read()
c1 = con.split('\n')
if c1[-1]=='':
    c1 = c1[:-1]
c1 = ['datetime'] + c1

store = pd.HDFStore(excel_path)
keys = store.keys()
print(keys)

for key in keys:
    df = store.get(key)
    print(f'key:{key}')
    df = df[c1]
    print(df)
    # with pd.ExcelWriter('./output.xlsx') as writer:
        # df.to_excel(writer, sheet_name='key')
    if os.path.exists(save_path):
        print('file exist.')
        df.to_hdf(save_path, key=key, mode='r+', format='table')
    else:
        print('file not exist.')
        df.to_hdf(save_path, key=key, mode='w', format='table')

