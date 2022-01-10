import pandas as pd
import numpy as np
import os
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

excel_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2020长三角.h5'
# save_path = excel_path.split('.h5')[0]+'长三角.h5'
# print(save_path)
# f = open('./长三角站点编码.txt', 'r')
# con = f.read()
# c1 = con.split('\n')
# c1 = ['datetime'] + c1

store = pd.HDFStore(excel_path)
keys = store.keys()
# print(keys)
print(excel_path)
mp = 0.1
no_site = set()
print(f'mp={mp}')
for key in keys:
    df = store.get(key)
    missing = draw_missing_data_table(df)
    cur_no_site = missing[missing['Percent']>mp].index.tolist()
    # print(type(cur_no_site), cur_no_site)
    print(key, len(cur_no_site))
    no_site = no_site.union(set(cur_no_site))
    total_num = len(missing.index)
    miss_num = len(missing[missing['Percent']>mp].index)
    miss_per = miss_num/total_num
    # print(missing)
    # print(missing.describe())
    # print(f'key:{key}, missing_percent:{mp}, total_num:{total_num}, num:{miss_num}, per:{miss_per:.4f}')
print('no_site', len(no_site))
# all_site = np.loadtxt('./长三角站点编码.txt')
f = open('./长三角站点编码.txt', 'r')
all_site = f.read().split('\n')
f.close()
all_site = set(all_site)
remain_site = all_site - no_site
remain_num = len(remain_site)
remain_site = np.array(list(remain_site))
# no_site = np.array(list(no_site))
np.savetxt(f'./长三角站点编码mp{mp}_len{remain_num}.txt', remain_site, fmt='%s', delimiter=',')