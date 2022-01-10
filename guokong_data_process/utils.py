import os 
import pandas as pd
import numpy as np
import argparse

def save_hdf2file(df, key, save_path, save_dir=None):
    if not save_dir is None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if os.path.exists(save_path):
        print(f'save to:{save_path}, key:{key}, r+')
        df.to_hdf(save_path, key=key, mode='r+', format='table')
    else:
        print(f'save to:{save_path}, key:{key}, w')
        df.to_hdf(save_path, key=key, mode='w', format='table')

def eliminate_df_column_slash(df):
    columns = df.columns
    df_ = df.copy()
    for old_name in columns:
        if old_name[0] == '/':
            new_name = old_name[1:]
            df_.rename(columns={old_name: new_name}, inplace=True)
    return df_

def restrain_df_by_stat(df, stat_df, key, metro_list=None):
    stat_df = eliminate_df_column_slash(stat_df)
    if metro_list is None:
        metro_list = ['cc', 'crwc', 'q', 't', 'u', 'v', 'uv_speed', 'uv_dir']
        metro_list_ = ['/cc', '/crwc', '/q', '/t', '/u', '/v', '/uv_speed', '/uv_dir']
    else:
        metro_list_ = [metro+'_' for metro in metro_list]
    metro_flag = False
    if key in metro_list or key in metro_list_:
        if key[0] == '/':
            key = key[1:]
        metro_flag = True
    else:
        key = [stat_df_column for stat_df_column in stat_df.columns if key in stat_df_column][0]
    u_max = stat_df.loc['u_max', key]
    u_min = stat_df.loc['u_min', key]
    u_nmax = stat_df.loc['u_nmax', key]
    u_nmin = stat_df.loc['u_nmin', key]
    # print(f'key:{key}, u_nmax:{u_nmax}, u_nmin:{u_nmin}, df max:{np.max(df.values)}, min:{np.min(df.values)}')
    cur_df = df.copy()
    if metro_flag:
        cur_df.loc[:, :] = cur_df.values * (u_max - u_min) + u_min
    else:
        cur_df.loc[:, :] = np.expm1(cur_df.values * (u_nmax - u_nmin) + u_nmin)
    return cur_df

def average_df_by_month(df):
    return df.set_index('datetime').groupby(pd.Grouper(freq='1D')).mean().reset_index().rename(columns={'index':'datetime'})

def see_df(hdf_path):
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    print(keys)
    for key in keys:
        print(key)
        df = store.get(key)
        print(df)

def get_season_df(total_df, season_name, year, season_dic=None):
    # season_list = ['summer', 'winter']
    if season_dic is None:
        season_dic = {
            'summer':[[0, 6], [0, 7], [0, 8]],
            'winter':[[0, 12], [1, 1], [1, 2]],
        }
    season_month_list = season_dic[season_name]
    ret_df = pd.DataFrame()
    for season_month_idx in range(len(season_month_list)):
        year_offset = season_month_list[season_month_idx][0]
        month_require = season_month_list[season_month_idx][1]
        temp_ret_df = total_df.loc[total_df['datetime'].dt.year==year+year_offset]
        temp_ret_df = temp_ret_df.loc[temp_ret_df['datetime'].dt.month==month_require]
        ret_df = pd.concat([ret_df, temp_ret_df], axis=0)
    return ret_df

def get_season_df_per_month_generator(total_df, season_name, year, new_freq='2h'):
    season_dic = {
        'summer':[[0, 6], [0, 7], [0, 8]],
        'winter':[[0, 12], [1, 1], [1, 2]],
    }
    season_month_list = season_dic[season_name]
    ret_df = pd.DataFrame()
    for season_month_idx in range(len(season_month_list)):
        year_offset = season_month_list[season_month_idx][0]
        month_require = season_month_list[season_month_idx][1]
        temp_ret_df = total_df.loc[total_df['datetime'].dt.year==year+year_offset]
        temp_ret_df = temp_ret_df.loc[temp_ret_df['datetime'].dt.month==month_require]
        new_ret_df = pd.DataFrame()
        new_ret_df['datetime'] = pd.date_range(temp_ret_df['datetime'].values[0], periods=len(temp_ret_df.index)//2, freq=new_freq)
        new_ret_df = pd.merge(left=new_ret_df, right=temp_ret_df, how='left')
        # print(f'temp_ret_df:\n{temp_ret_df}\nnew_ret_df:\n{new_ret_df}')
        yield new_ret_df

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')