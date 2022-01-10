from calendar import monthrange
import pandas as pd
import numpy as np
import os 

def create_year_h5(year, save_name, monitor_data_dir_path,monitor_data_path_bone,  site_list, category_list):
    cat_dic = {}
    for single_cat in category_list:
        cat_dic[single_cat] = pd.DataFrame(columns=['datetime']+site_list)
    for month in range(1,13):
        print(f'month:{month}')
        max_day = int(monthrange(year, month)[1])
        for day in range(1, max_day+1):
            valid_input_flag = False
            monitor_data_path = monitor_data_dir_path + monitor_data_path_bone.format(year, str(month).zfill(2), str(day).zfill(2))
            if os.path.exists(monitor_data_path):
                monitor_data_df = pd.read_csv(monitor_data_path)
                if 'date' in monitor_data_df.columns:
                    column_empty_df = pd.DataFrame(columns=['datetime', 'type']+site_list)
                    monitor_data_df['datetime'] = pd.to_datetime(monitor_data_df['date'].astype(str) + monitor_data_df['hour'].astype(str), format='%Y%m%d%H')
                    valid_input_flag = True
                    monitor_data_df = monitor_data_df[monitor_data_df.columns.intersection(['datetime', 'type']+site_list)]
                    monitor_data_df = pd.concat([column_empty_df, monitor_data_df], axis=0, ignore_index=True)
                                
            start_time = f'{year}-{month}-{day} 00:00:00'
            end_time = f'{year}-{month}-{day} 23:00:00'
            time_index = pd.date_range(start_time, end_time, freq='1h')
            empty_df = pd.DataFrame()
            empty_df['datetime'] = time_index
            for single_cat in category_list:
                cat_df = cat_dic[single_cat]
                if valid_input_flag:
                    cur_df = monitor_data_df[monitor_data_df['type']==single_cat]
                    cur_df = cur_df.drop(columns=['type'])
                    cur_df = pd.merge(empty_df, cur_df, how='left', on=['datetime'])
                else:
                    cur_df = pd.DataFrame(columns=site_list)
                    cur_df['datetime'] = time_index
                    cur_df = cur_df[['datetime']+site_list]
                cat_df = pd.concat([cat_df, cur_df], axis=0, ignore_index=True)
                cat_dic[single_cat] = cat_df
                
    for key, value in cat_dic.items():
        # print(f'key:{key}')
        # print(f'value:{value}')
        # print(f'columns:{value.columns}')
        # print(f'dtypes:{value.dtypes}')
        value[value.columns[1:]] = value[value.columns[1:]].astype('float64')
        # for i, v in value.dtypes.items():
            # print(f'index: {i}, value:{v}')
            # if v == 'object':
                # print(value[i])
                # for c_idx in range(len(value[i])):
                    # print(value[i].iloc[c_idx])
        if os.path.exists(save_name):
            print(f'save to:{save_name}, key:{key}, r+')
            value.to_hdf(save_name, key=key, mode='r+', format='table')
        else:
            print(f'save to:{save_name}, key:{key}, w')
            value.to_hdf(save_name, key=key, mode='w', format='table')

def main():
    year_list = range(2015, 2013, -1)
    for year in year_list:
    # year = 2018
        print(f'year:{year}')
        suffix = '长三角'
        save_path = f'/mnt/d/codes/downloads/datasets/国控站点/h5/guokong{year}{suffix}.h5'
        monitor_data_dir_path = f'/mnt/d/codes/downloads/datasets/国控站点/站点_{year}0101-{year}1231/'
        monitor_data_path_bone = 'china_sites_{}{}{}.csv'
        site_list_path = '../长三角站点编码mp0.05_len176.txt'
        with open(site_list_path, 'r') as f:
            lines = f.readlines()
            site_list = [s[:-1] for s in lines]
            # print(site_list) 
        category_list = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
        create_year_h5(year, save_path, monitor_data_dir_path, monitor_data_path_bone, site_list, category_list)
if __name__ == '__main__':
    main()