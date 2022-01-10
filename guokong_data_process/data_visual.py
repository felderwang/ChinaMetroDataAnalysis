import geopandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import guokong_data_process
import math

def draw_series(timeindexs, series, colors, labels, alphas, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = save_dir + save_name
    fig, axs = plt.subplots(1,1, figsize=(80, 4))
    ax = axs
    
    for idx in range(len(series)):
        ax.plot(timeindexs[idx], series[idx], color=colors[idx], label=labels[idx], alpha=alphas[idx])
    plt.legend()
    print(f'save to: {save_name}')
    plt.savefig(save_name)
    plt.clf()
    plt.close()

def draw_source_res_series(source_path, res_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    source_color = '#FF0000'
    res_color = '#0000FF'
    source_label = 'source'
    res_label = 'res'
    source_alpha = 0.7
    res_alpha = 0.7

    colors = [source_color, res_color]
    labels = [source_label, res_label]
    alphas = [source_alpha, res_alpha]

    source_store = pd.HDFStore(source_path)
    res_store = pd.HDFStore(res_path)
    res_keys = res_store.keys()

    for res_key in res_keys:
        source_df = source_store.get(res_key)
        res_df = res_store.get(res_key)
        if res_key[0]=='/':
            res_key = res_key[1:]
        timeindex = source_df.iloc[:,0]
        timeindexs = [timeindex, timeindex]
        res_columns = res_df.columns[1:]
        for res_column in res_columns:
            source_series = source_df[res_column]
            res_series = res_df[res_column]

            source_mask = source_series == 0
            source_series[source_mask] = np.nan
            res_series[np.logical_not(source_mask)] = np.nan

            series = [source_series, res_series]
            cur_save_dir = save_dir + f'{res_key}/'
            save_name = f'{res_key}_{res_column}_sourceres.png'
            draw_series(timeindexs, series, colors, labels, alphas, cur_save_dir, save_name)            

def draw_sites_with_map(site_txt_path, site_path, geo_path, save_dir, save_name, interval=0.25):
    '''
    经度 longitude
    纬度 latitude
    '''
    color_site = '#ff0000'
    color_line = '#00ff00'
    with open(site_txt_path, 'r') as f:
        lines = f.readlines()
        site_list = [s[:-1] for s in lines]
    df_site = pd.read_excel(site_path)
    df_site = df_site[df_site['监测点编码'].isin(site_list)]
    print(df_site)
    lons = df_site['经度'].values
    lats = df_site['纬度'].values
    min_lon = math.floor(np.min(lons))
    max_lon = math.ceil(np.max(lons))
    min_lat = math.floor(np.min(lats))
    max_lat = math.ceil(np.max(lats))

    states = geopandas.read_file(geo_path)
    # fig, axs = plt.subplots(1,1, figsize=((max_lon-min_lon)*5,(max_lat-min_lat)*5))
    fig, axs = plt.subplots(1,1, figsize=(200,(max_lat-min_lat)*5))
    ax = axs
    ax.title.set_text(save_name.split('.')[0])
    ax.scatter(lons, lats, color=color_site)
    states.boundary.plot(ax=ax)
    
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    gap = 0.25
    for y_idx in np.arange(min_lat, max_lat, gap):
        y = [y_idx, y_idx]
        x = [min_lon, max_lon]
        ax.plot(x, y, color=color_line)
    for x_idx in np.arange(min_lon, max_lon, gap):
        x = [x_idx, x_idx]
        y = [min_lat, max_lat]
        ax.plot(x, y, color=color_line)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir + save_name
    print(f'save to:{save_path}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def transfer_result(transfer_hdf_dir, transfer_hdf_name, save_dir, year_list):
    '''
    # 大标题：A2B，
    小标题：年份，
    里面各个站点画在一起
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    transfer_hdf_path = transfer_hdf_dir + transfer_hdf_name
    transfer_store = pd.HDFStore(transfer_hdf_path)
    keys = transfer_store.keys()
    
    print(keys)
    for transfer_key in keys:
        # save_name = f'{transfer_key}_{year_list[0]}_{year_list[-1]}.png'
        save_name = f'1159A/{transfer_key}_{year_list[0]}_{year_list[-1]}_1159A.png'
        save_path = save_dir + save_name
        fig, axs = plt.subplots(1,len(year_list), figsize=(10*len(year_list)*2, 10))
        fig.suptitle(f'transfer_key:{transfer_key}')
        axs = axs.flat
        transfer_df = transfer_store.get(transfer_key)
        for i, year in enumerate(year_list):
            year_df = transfer_df[transfer_df.index.str.contains(str(year))]
            len_year_df = len(year_df)
            colors = cm.rainbow(np.linspace(0, 1, len(year_df.columns)))
            ax = axs[i]
            ax.title.set_text(f'year:{year}, transfer:{transfer_key}')
            ax.set_xlabel(f'lags (day)')
            ax.set_ylabel(f'transfer entropy')
            # for site_idx, site in enumerate(year_df.columns):
            for site_idx, site in enumerate(['1159A']):
                ax.plot(np.arange(1, len_year_df+1, 1), year_df[site].values, color=colors[site_idx])
        # plt.legend(loc='best')
        plt.legend()
        print(f'save to:{save_path}')
        plt.savefig(save_path)
        plt.clf()
        plt.close()


def main():
    guokong_res_hdf_path = f'/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545.h5'
    guokong_source_hdf_path = f'/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2014_2020长三角176log_exclude3std_reshape2n01_notstlplus.h5'

    source_res_fig_save_dir = '../pics/source_res/2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545/'
    # draw_source_res_series(guokong_source_hdf_path, guokong_res_hdf_path, source_res_fig_save_dir)
    gkcsj140_site_txt_path = '/mnt/d/codes/downloads/datasets/国控站点/新站点列表/csj140.txt'
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    china_geo_path = '/mnt/d/codes/downloads/datasets/国控站点/cnmap/CHN_adm0.shp'
    gkcsj140_sites_pic_save_dir = '../pics/'
    gkcsj140_sites_pic_save_name = 'gkcsj140_sites.png'

    transfer_hdf_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    transfer_hdf_name = 'transfer_entropy2016_2020长三角140_day.h5'
    season_summer_transfer_hdf_name = 'transfer_entropy2016_2020长三角140_day_summer.h5'
    season_winter_transfer_hdf_name = 'transfer_entropy2016_2020长三角140_day_winter.h5'
    year_list = range(2016, 2021)
    season_year_list = range(2018, 2020)
    transfer_pic_save_dir = f'../pics/transfer_result/'
    season_summer_transfer_pic_save_dir = f'../pics/transfer_result/summer/'
    season_winter_transfer_pic_save_dir = f'../pics/transfer_result/winter/'
    # guokong_data_process.see_df(transfer_hdf_dir+transfer_hdf_name)
    # draw_sites_with_map(gkcsj140_site_txt_path, site_path, china_geo_path, gkcsj140_sites_pic_save_dir, gkcsj140_sites_pic_save_name)
    # transfer_result(transfer_hdf_dir, transfer_hdf_name, transfer_pic_save_dir, year_list)
    # transfer_result(transfer_hdf_dir, season_summer_transfer_hdf_name, season_summer_transfer_pic_save_dir, season_year_list)
    # transfer_result(transfer_hdf_dir, season_winter_transfer_hdf_name, season_winter_transfer_pic_save_dir, season_year_list)
    guokong_data_process.see_df(transfer_hdf_dir+season_summer_transfer_hdf_name)

if __name__ == '__main__':
    main()