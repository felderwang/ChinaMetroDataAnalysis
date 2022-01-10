import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix

excel_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2020长三角mp0.05_len176linearinter.h5'

a_file = open('./site_dic.pkl', 'rb')
site_dic = pickle.load(a_file)
# print(site_dic)

store = pd.HDFStore(excel_path)
keys = store.keys()
key = keys[0]

df = store.get(key)
store.close()
# print(df)
df.drop(columns=['datetime'], inplace=True)
col = df.columns.values
np_ar = np.zeros((len(col),2))
print(f'np_ar:{np_ar.shape}')
# print(col)
for idx, site in enumerate(col):
    site_loc = site_dic[site]
    np_ar[idx, 0] = site_loc[0]
    np_ar[idx, 1] = site_loc[1]
print(np_ar, np_ar.shape)
# np_ar = np_ar.tolist()

dis_mat = distance_matrix(np_ar, np_ar, p=2)
print(dis_mat, dis_mat.shape)