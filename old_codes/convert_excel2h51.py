from calendar import monthrange
import pandas as pd
import numpy as np
import collections
import os
# def monitor_data_path(bone='china_sites_{}{}{}',year, month, day):
    # return 



def create_year_h5(year, site_list, monitor_data_path_bone, monitor_data_dir_path, category_list, h5_save_path):
    category_dic = {}
    for category in category_list:
        category_dic[category] = pd.DataFrame(columns=site_list)
    h5_save_name = h5_save_path+f'guokong{year}.h5'
    for month in range(1,13):
        max_day = int(monthrange(year, month)[1])
        print(f'month:{month}, max_day:{max_day}')
        for day in range(1, max_day+1):
            monitor_data_path = monitor_data_dir_path + monitor_data_path_bone.format(year, str(month).zfill(2), str(day).zfill(2))
            # print(monitor_data_path)
            monitor_data_df = pd.read_csv(monitor_data_path)

            monitor_data_df['datetime'] = pd.to_datetime(monitor_data_df['date'].astype(str) + monitor_data_df['hour'].astype(str), format='%Y%m%d%H')
            monitor_data_df.drop(columns=['date', 'hour'], inplace=True)
            for category in category_list:
                # if not category in category_dic:
                    # category_dic[category] = pd.DataFrame(columns=site_list)
                category_df = category_dic[category]
                cur_df = monitor_data_df.loc[monitor_data_df['type']==category]
                cur_df = cur_df.drop(columns=['type'])
                category_df = category_df.append(cur_df, sort=False, ignore_index=True)
                category_dic[category] = category_df
    for key, value in category_dic.items():
        print(f'key:{key}')
        print(f'value:{value}')
        if os.path.exists(h5_save_name):
            print(f'file exist')
            value.to_hdf(h5_save_name, key=key, mode='r+', format='table')
        else:
            print(f'file not exist')
            value.to_hdf(h5_save_name, key=key, mode='w', format='table')



def main():
    year = 2017
    category_list = ['AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
    site_list_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/'
    latest_site_path = site_list_path+'站点列表-2021.01.01起.xlsx'
    latest_site = pd.read_excel(latest_site_path)
    site_list_dup = latest_site['监测点编码'].values
    site_list_dup = ['datetime']+site_list_dup.tolist()#dup 1961A, 1966A
    site_list = []
    [site_list.append(x) for x in site_list_dup if x not in site_list]

    monitor_data_fdir_path = '/mnt/d/codes/downloads/datasets/国控站点/'
    
    monitor_data_dir_path = f'{monitor_data_fdir_path}站点_{year}0101-{year}1231/'
    monitor_data_path_bone = 'china_sites_{}{}{}.csv'

    h5_save_path = monitor_data_fdir_path+'h5/'

    if not os.path.exists(h5_save_path):
        os.makedirs(h5_save_path)
    
    create_year_h5(year, site_list, monitor_data_path_bone, monitor_data_dir_path,category_list, h5_save_path)

if __name__ == '__main__':
    main()