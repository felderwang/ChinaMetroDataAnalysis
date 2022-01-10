import pandas as pd
import numpy as np
import collections

# category_list = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
category_list = ['AQI', 'PM2.5']
category_dic={}
site_list_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/'

latest_site_path = site_list_path+'站点列表-2021.01.01起.xlsx'

latest_site = pd.read_excel(latest_site_path)

# print(latest_site)            
site_list_dup = latest_site['监测点编码'].values
site_list_dup = site_list_dup.tolist()+['datetime']#dup 1961A, 1966A
site_list = []
[site_list.append(x) for x in site_list_dup if x not in site_list]
print(f'site_dup:{len(site_list_dup)}, site:{len(site_list)}')
# print(site_list.shape)#length 2024+1

result_df = pd.DataFrame(columns=site_list)
# print(result_df)

monitor_data_path = '/mnt/d/codes/downloads/datasets/国控站点/站点_20190101-20191231/china_sites_20190501.csv'
# monitor_data_path = '/mnt/d/codes/downloads/datasets/国控站点/站点_20210101-20210327/china_sites_20210101.csv'

monitor_data_df = pd.read_csv(monitor_data_path)

monitor_data_df['datetime'] = pd.to_datetime(monitor_data_df['date'].astype(str) + monitor_data_df['hour'].astype(str), format='%Y%m%d%H')
# print(monitor_data_df.head())
# monitor_data_df.set_index(['datetime'], inplace=True)
# monitor_data_df.index.name=None
# print(monitor_data_df['datetime'])
monitor_data_df.drop(columns=['date', 'hour'], inplace=True)
# monitor_data_df = monitor_data_df.reset_index(monitor_data_df['datetime'])
# print(monitor_data_df.head())
# print(monitor_data_df['type'].unique())
# print(monitor_data_df.info())

for category in category_list:
    # globals()[f'{category}_df']=pd.DataFrame(columns=site_list)
    if not category in category_dic:
        category_dic[category] = pd.DataFrame(columns=site_list)
    category_df = category_dic[category]
    cur_df = monitor_data_df.loc[monitor_data_df['type']==category]
    cur_df = cur_df.drop(columns=['type'])
    print(category)
    print(cur_df.head())
    # print(category_df.head())
    category_df = category_df.append(cur_df, sort=False, ignore_index=True)
    print(category_df.head())
# for category in category_list:
    # print(category)
# result_df = result_df.append(monitor_data_df, sort=False)
# print(result.head())
