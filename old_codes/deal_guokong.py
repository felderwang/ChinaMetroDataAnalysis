import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import random
SEED = 114514

random.seed(SEED)
np.random.seed(SEED)

def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def closest_node(node, nodes):
    # nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    # return np.argmin(dist_2)
    result = np.argpartition(dist_2, 2)
    # print(result[1])
    return result[1]
def deal_guokong(h5_path, select_list=None, year=2020, step='1h', site_path=None, site_txt=None, mp=0.1):
    # save_path = '/home/why/Document/datasets/guokong/h5/guokong2019长三角mp0.05_len176_year_inter.h5'
    save_path = h5_path.split('.h5')[0]+'_year_inter.h5'
    store = pd.HDFStore(h5_path)
    keys = store.keys()
    if select_list is not None:
        keys = select_list
    print(keys)
    if site_path is not None and site_txt is not None:
        with open(site_txt, 'r') as f:
            lines = f.readlines()
            site_list = [s[:-1] for s in lines]
            print(site_list)
        df_site = pd.read_excel(site_path)
        df_site = df_site[df_site['监测点编码'].isin(site_list)]
        df_site = df_site.set_index('监测点编码')
        df_site = df_site.T
        df_site = df_site[site_list]
        lon_lat_ar = np.asarray(df_site.loc[['经度','纬度']].values).T
        print(f'lon_lat_ar:{lon_lat_ar.shape}')
    for key in keys:
        print(f'key:{key}')
        df = store.get(key)
        time_index = pd.date_range(f'1/1/{str(year)}', f'1/1/{str(int(year)+1)}', freq=step)
        time_index = time_index[:-1]
        new_df = pd.DataFrame()
        new_df['datetime'] = time_index
        new_df = pd.merge(new_df, df, on='datetime', how='left')
        missing = draw_missing_data_table(df)
        cur_no_site = missing[missing['Percent']>mp].index.tolist()
        print(f'cur_no_site:{cur_no_site}')
        for no_site in cur_no_site:
            no_site_index = site_list.index(no_site)
            print(f'no_site:{no_site}, index:{no_site_index}, in list:{site_list[no_site_index]}')
            closet_site_index = closest_node(lon_lat_ar[no_site_index], lon_lat_ar)
            print(f'closet_site_index:{site_list[closet_site_index]}, lon lat:{lon_lat_ar[closet_site_index]}')
            new_df[no_site] = new_df[site_list[closet_site_index]]
        # assert(0)
        # new_df.iloc[0] = new_df.iloc[0].fillna(0.0)
        # new_df.interpolate(inplace=True)
        
        columns = new_df.columns[1:]
        print(new_df)
        nan_num = new_df.isnull().sum().sum()
        print(f'nan_num:{nan_num}')
        if os.path.exists(save_path):
            print(f'file exist. key:{key}')
            new_df.to_hdf(save_path, key=key, mode='r+', format='table')
        else:
            print(f'file not exist. key:{key}, save_path:{save_path}')
            new_df.to_hdf(save_path, key=key, mode='w', format='table')


def draw_series(h5_path, select_list=None, year=2020, step='1h'):
    # save_path = '/home/why/Document/datasets/guokong/h5/guokong2019长三角mp0.05_len176_year_inter.h5'
    save_path = h5_path.split('.h5')[0]+'_year_inter.h5'
    store = pd.HDFStore(h5_path)
    keys = store.keys()
    if select_list is not None:
        keys = select_list
    print(keys)
    for key in keys:
        print(f'key:{key}')
        df = store.get(key)
        df['datetime'] = df['datetime'].dt.floor('h')
        sr_timeindex = df['datetime']
        sr_timeindex_dup = sr_timeindex.duplicated()
        dup_index = sr_timeindex_dup[sr_timeindex_dup].index
        df = df.drop(dup_index)
        df = df.reset_index()
        df.drop(columns=['index'], inplace=True)
        new_df = df
        time_index = pd.date_range(f'1/1/{str(year)}', f'1/1/{str(int(year)+1)}', freq=step)
        time_index = time_index[:-1]
        # new_df = pd.DataFrame()
        # new_df['datetime'] = time_index
        # new_df = pd.merge(new_df, df, on='datetime', how='left')
        # new_df.iloc[0] = new_df.iloc[0].fillna(0.0)
        # new_df.interpolate(inplace=True)
        columns = new_df.columns[1:]
        print(new_df)
        nan_num = new_df.isnull().sum().sum()
        print(f'nan_num:{nan_num}')
        # if os.path.exists(save_path):
        #     print(f'file exist. key:{key}')
        #     new_df.to_hdf(save_path, key=key, mode='r+', format='table')
        # else:
        #     print(f'file not exist. key:{key}, save_path:{save_path}')
        #     new_df.to_hdf(save_path, key=key, mode='w', format='table')
        fig, axs = plt.subplots(1,1, figsize=(50,8))
        ax = axs
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(columns)))
        for idx, column in enumerate(columns):
            column_value = new_df[column].values
            ax.plot(time_index, column_value, color=colors[idx], label=column, linewidth=0.5, alpha=0.5)
        plt.legend()
        save_dir = f'./pics/guokong{year}/total/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = save_dir + f'{key}_{year}_total.png'
        plt.savefig(file_name)
        print(f'save to: {file_name}')
        plt.clf()
        plt.close()

def draw_series_single(h5_path, select_list=None, year=2020, step='1h'):
    store = pd.HDFStore(h5_path)
    keys = store.keys()
    if select_list is not None:
        keys = select_list
    print(keys)

    for key in keys:
        print(f'key:{key}')
        if key[0] == '/':
            key = key[1:]
        df = store.get(key)
        df['datetime'] = df['datetime'].dt.floor('h')
        sr_timeindex = df['datetime']
        sr_timeindex_dup = sr_timeindex.duplicated()
        dup_index = sr_timeindex_dup[sr_timeindex_dup].index
        # print(f'dup_index:{dup_index}')
        # print(sr_timeindex.iloc[dup_index])
        # print(df.iloc[dup_index])
        df = df.drop(dup_index)
        df = df.reset_index()
        df.drop(columns=['index'], inplace=True)
        time_index = pd.date_range(f'1/1/{str(year)}', f'1/1/{str(int(year)+1)}', freq=step)
        time_index = time_index[:-1]
        new_df = pd.DataFrame()
        new_df['datetime'] = time_index
        new_df = pd.merge(new_df, df, on='datetime', how='left')
        columns = new_df.columns[1:]
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(columns)))
        color='#ff0000'
        for idx, column in enumerate(columns):
            fig, axs = plt.subplots(1,1, figsize=(50,8))
            ax = axs
            column_value = new_df[column].values
            print(f'len time_index:{len(time_index)}, column:{column} {len(column_value)}')
            ax.plot(time_index, column_value, color=color, label=column, , alpha=1)
            plt.legend()
            save_dir = f'./pics/guokong{year}/{key}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = save_dir + f'{column}_{year}_{key}.png'
            plt.savefig(file_name)
            print(f'save to: {file_name}')
            plt.clf()
            plt.close()
def draw_series_single_bin(h5_path, select_list=None, year=2020, step='1h', bins=500):
    store = pd.HDFStore(h5_path)
    keys = store.keys()
    if select_list is not None:
        keys = select_list
    print(keys)
    for key in keys:
        print(f'key:{key}')
        if key[0] == '/':
            key = key[1:]
        df = store.get(key)
        df['datetime'] = df['datetime'].dt.floor('h')
        sr_timeindex = df['datetime']
        sr_timeindex_dup = sr_timeindex.duplicated()
        dup_index = sr_timeindex_dup[sr_timeindex_dup].index
        # print(f'dup_index:{dup_index}')
        # print(sr_timeindex.iloc[dup_index])
        # print(df.iloc[dup_index])
        df = df.drop(dup_index)
        df = df.reset_index()
        df.drop(columns=['index'], inplace=True)
        time_index = pd.date_range(f'1/1/{str(year)}', f'1/1/{str(int(year)+1)}', freq=step)
        time_index = time_index[:-1]
        new_df = pd.DataFrame()
        new_df['datetime'] = time_index
        new_df = pd.merge(new_df, df, on='datetime', how='left')
        columns = new_df.columns[1:]
        
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(columns)))
        for idx, column in enumerate(columns):
            fig, axs = plt.subplots(1,1, figsize=(8,8))
            ax = axs
            column_value = new_df[column].values
            column_value = column_value[~np.isnan(column_value)]
            if len(column_value) == 0:
                print(f'len 0.')
                assert(0)
            mean = np.mean(column_value)
            std = np.std(column_value)

            y, x, _ = ax.hist(column_value, bins=bins)
            x_left = mean - 3*std
            x_lleft = mean - 10*std
            x_right = mean + 3*std
            x_rright = mean + 10*std
            y_max = y.max()
            x_max = x.max()
            elem = np.where(y == y_max)
            ax.plot([x_left, x_left], [0, y_max])
            ax.plot([x_lleft, x_lleft], [0, y_max])
            ax.plot([x_right, x_right], [0, y_max])
            ax.plot([x_rright, x_rright], [0, y_max])
            
            ax.title.set_text(f'{column}: year:{year}\nmean:{mean:.2f}, std:{std:.2f}, x_max:{x[elem][0]}, y_max:{y.max()}')

            plt.legend()
            save_dir = f'./pics/guokong{year}/bins/{key}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = save_dir + f'bins_{column}_{year}_{key}.png'
            plt.savefig(file_name)
            print(f'save to: {file_name}')
            plt.clf()
            plt.close()


def main():
    guokong_h5_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2019长三角mp0.05_len176_year_inter.h5'
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    site_txt = './长三角站点编码mp0.05_len176.txt'
    key_list = ['/CO', '/NO2', '/O3', '/PM10', '/PM2.5', '/SO2']
    # draw_series(guokong_h5_path, key_list, year=2019)
    # deal_guokong(guokong_h5_path, key_list, year=2019, site_path=site_path, site_txt=site_txt)
    
    draw_series_single(guokong_h5_path, key_list, year=2019)
    # draw_series_single_bin(guokong_h5_path, key_list, year=2019)

if __name__ == '__main__':
    main()