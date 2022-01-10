import pandas as pd
import numpy as np
import collections

excel_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2020_m1.h5'

store = pd.HDFStore(excel_path)
keys = store.keys()
print(keys)

for key in keys:
    df = store.get(key)
    print(f'key:{key}')
    print(df)
    # with pd.ExcelWriter('./output.xlsx') as writer:
        # df.to_excel(writer, sheet_name='key')
    