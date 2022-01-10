import pandas as pd
import numpy as np
import os
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

h5_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2020长三角mp0.05_len176_year_inter.h5'
store = pd.HDFStore(h5_path)
keys = store.keys()

for key in keys:
    print(key)
    df = store.get(key)
    columns = df.columns
    print(columns)
    # print(f'len columns:{len(columns)}')
    # missing = draw_missing_data_table(df)
    # print(missing)
    # cur_no_site = missing[missing['Percent']>0.05].index.tolist()
    # print(f'len cur_no_site:{len(cur_no_site)}\n{cur_no_site}')
    # for no_site in cur_no_site:
    #     print(f'{no_site}, {missing.loc[no_site]["Percent"]}')
