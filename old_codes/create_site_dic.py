import pandas as pd
import numpy as np
import pickle

site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/' + '站点列表-2021.01.01起.xlsx'
latest_site = pd.read_excel(site_path)
site_list_dup = latest_site['监测点编码'].values
site_dic = {}
# site_list = []
# [site_list.append(x) for x in site_list_dup if x not in site_list]
for i in range(0, len(latest_site.index)):
    site = latest_site.loc[i, '监测点编码']
    lat = latest_site.loc[i, '纬度']
    lon = latest_site.loc[i, '经度']
    site_dic[site]=[lat, lon]
    # print(f'site:{site}, {type(site)}')
# print(site_dic)
a_file = open('./site_dic.pkl', 'wb')
pickle.dump(site_dic, a_file)
a_file.close()