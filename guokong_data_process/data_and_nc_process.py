import pandas as pd
import numpy as np
import os
import netCDF4
import utils
# from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# import rpy2.robjects as robjects
# numpy2ri.activate()
import copent
import argparse
from tqdm import tqdm



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_pollute', type=utils.str2bool, default=False)
    parser.add_argument('--pollute_idx', type=int, default=0)
    args = parser.parse_args()
    return args

def find_nearest_idx(ar, value):
    ar = np.asarray(ar)
    idx = (np.abs(ar-value)).argmin()
    # print(f'ar:{ar}\nvalue:{value}, idx:{idx}')
    return idx

def convert_list2txt(site_list, save_dir, save_name):
    save_path = save_dir + save_name
    textfile = open(save_path, 'w')
    for site in site_list:
        textfile.write(site+'\n')
    textfile.close()

def see_nc(nc_dir, nc_name):
    nc_path = nc_dir + nc_name
    nc_obj = netCDF4.Dataset(nc_path)
    keys = nc_obj.variables.keys()
    for key in list(keys):
        print(key)
        if key == 'level' or key == 'longitude' or key == 'latitude':
            print(nc_obj.variables[key][:])

def generate_nc2site_hdf(nc_dir, nc_name, lon_index, lat_index, site_path, site_txt, nc_data_list, save_dir , save_name, year_list, level=-1, source_step='6h', target_step='1h'):
    save_path = save_dir + save_name
    
    stat_name = ['u_mean', 'u_std', 'u_max', 'u_min', 'u_nmean', 'u_nstd', 'u_nmax', 'u_nmin']

    stat_df = pd.DataFrame()
    stat_df['stat_name'] = stat_name

    nc_df = pd.DataFrame()
    nc_df['datetime'] = pd.date_range(f'1/1/{str(year_list[0])}', f'1/1/{str(int(year_list[-1])+1)}', freq=source_step)[:-1]

    target_df = pd.DataFrame()
    target_df['datetime'] = pd.date_range(f'1/1/{str(year_list[0])}', f'1/1/{str(int(year_list[-1])+1)}', freq=target_step)[:-1]

    nc_data_df_dic = {}
    target_data_df_dic = {}
    stat_dic = {}
    for nc_data_cat in nc_data_list:
        nc_data_df_dic[nc_data_cat] = nc_df.copy()
        target_data_df_dic[nc_data_cat] = target_df.copy()

    with open(site_txt, 'r') as f:
        lines = f.readlines()
        site_list = [s[:-1] for s in lines]

    df_site = pd.read_excel(site_path)
    df_site = df_site[df_site['监测点编码'].isin(site_list)]
    nc_data_dic = {}

    for year in year_list:
        nc_path = nc_dir + nc_name.format(year)
        print(f'nc_path:{nc_path}')
        nc_obj = netCDF4.Dataset(nc_path)
        keys = nc_obj.variables.keys()

        for key in list(keys):
            if key == 'longitude':
                data = np.array(nc_obj.variables[key][:])
                lon = data[lon_index]
            elif key == 'latitude':
                data = np.array(nc_obj.variables[key][:])
                lat = data[lat_index]
            elif key in nc_data_list:
                data = np.array(nc_obj.variables[key][:])
                data = data[:, level, lat_index[:, None], lon_index]
                
                if not key in nc_data_dic.keys():
                    nc_data_dic[key] = data
                else:
                    nc_data_dic[key] = np.concatenate((nc_data_dic[key], data), axis=0)
                # print(f'key:{key}, data:{nc_data_dic[key].shape}, single:{data.shape}')
    for nc_data_cat in nc_data_list:
        data = nc_data_dic[nc_data_cat]
        u_mean = np.mean(data)
        u_std = np.std(data)
        u_max = np.max(data)
        u_min = np.min(data)
        stat_dic[nc_data_cat] = [u_mean, u_std, u_max, u_min]
    #     u_nmax = np.max(data)
    #     u_nmin = np.min(data)

    #     data = (data-u_nmin)/(u_nmax-u_nmin)

    #     u_nmean = np.mean(data)
    #     u_nstd = np.std(data)
    #     nc_data_dic[nc_data_cat] = data
    #     stat_df[nc_data_cat] = [u_mean, u_std, u_max, u_min, u_nmean, u_nstd, u_nmax, u_nmin]

    stat_df.set_index('stat_name', inplace=True)
    
    for site in site_list:
        site_sr = df_site[df_site['监测点编码']==site]
        site_lon = site_sr['经度'].values[0]
        site_lat = site_sr['纬度'].values[0]
        closest_lat_idx = find_nearest_idx(lat, site_lat)
        closest_lon_idx = find_nearest_idx(lon, site_lon)
        for nc_data_cat in nc_data_list:
            df = nc_data_df_dic[nc_data_cat]
            nc_data = nc_data_dic[nc_data_cat]
            df[site] = nc_data[:, closest_lat_idx, closest_lon_idx]
            nc_data_df_dic[nc_data_cat] = df
    for nc_data_cat in nc_data_list:
        source_df = nc_data_df_dic[nc_data_cat]
        target_df = target_data_df_dic[nc_data_cat]
        target_df = pd.merge(target_df, source_df, on='datetime', how='left')
        target_df.interpolate(inplace=True)

        data = target_df[site_list].values
        u_mean, u_std, u_max, u_min = stat_dic[nc_data_cat]
        data = (data-u_max)/(u_max-u_min)
        u_nmax = np.max(data)
        u_nmin = np.min(data)
        u_nmean = np.mean(data)
        u_nstd = np.std(data)
        stat_df[nc_data_cat] = [u_mean, u_std, u_max, u_min, u_nmean, u_nstd, u_nmax, u_nmin]

    # print(stat_df)
        utils.save_hdf2file(target_df, nc_data_cat, save_path)
        # if os.path.exists(save_path):
            # print(f'save to:{save_path}, key:{nc_data_cat}, r+')
            # target_df.to_hdf(save_path, key=nc_data_cat, mode='r+', format='table')
        # else:
            # print(f'save to:{save_path}, key:{nc_data_cat}, w')
            # target_df.to_hdf(save_path, key=nc_data_cat, mode='w', format='table')
    # print(f'save to:{save_path}, key:{"stat"}, r+')
    # stat_df.to_hdf(save_path, key='stat', mode='r+', format='table')
    utils.save_hdf2file(stat_df, 'stat', save_path)

def combine_metro_and_nc_hdf(metro_hdf_dir, metro_hdf_name, nc_hdf_dir, nc_hdf_name, save_dir, save_name, stat_name = 'stat'):
    metro_hdf_path = metro_hdf_dir + metro_hdf_name
    nc_hdf_path = nc_hdf_dir + nc_hdf_name
    save_path = save_dir + save_name
    metro_store = pd.HDFStore(metro_hdf_path)
    nc_store = pd.HDFStore(nc_hdf_path)
    metro_store_keys = metro_store.keys()
    nc_store_keys = nc_store.keys()
    stat_df = None
    for metro_key in metro_store_keys:
        print(f'metro_key:{metro_key}')
        out_df = metro_store.get(metro_key)
        if metro_key[0] == '/':
            metro_key = metro_key[1:]

        if metro_key == stat_name:
            stat_df = out_df
        else:
            if os.path.exists(save_path):
                print(f'save to:{save_path}, key:{metro_key}, r+')
                out_df.to_hdf(save_path, key=metro_key, mode='r+', format='table')
            else:
                print(f'save to:{save_path}, key:{metro_key}, w')
                out_df.to_hdf(save_path, key=metro_key, mode='w', format='table')
    for nc_key in nc_store_keys:
        print(f'nc_key:{nc_key}')
        out_df = nc_store.get(nc_key)
        if nc_key[0] == '/':
            nc_key = nc_key[1:]
        if nc_key == stat_name:
            stat_df = pd.concat((stat_df, out_df), axis=1)
        else:
            print(f'save to:{save_path}, key:{nc_key}, r+')
            out_df.to_hdf(save_path, key=nc_key, mode='r+', format='table')

    print(stat_df)
    print(f'save to:{save_path}, key:{stat_name}, r+')
    stat_df.to_hdf(save_path, key=stat_name, mode='r+', format='table')



def convert_uv2speed_dir(source_dir, source_name, save_dir, save_name ,site_list):
    source_path = source_dir + source_name
    save_path = save_dir + save_name
    store = pd.HDFStore(source_path)
    stat_df = store.get('stat')
    u_df = store.get('u')
    v_df = store.get('v')
    u_df.loc[:, site_list] = utils.restrain_df_by_stat(u_df[site_list], stat_df, 'u')
    v_df.loc[:, site_list] = utils.restrain_df_by_stat(v_df[site_list], stat_df, 'v')
    #u_mean, u_std, u_max, u_min, u_nmean, u_nstd, u_nmax, u_nmin
    uv_dic = {}
    uv_speed_df = u_df.copy()
    uv_dir_df = u_df.copy()
    uv_speed_df.loc[:, site_list] = np.sqrt(u_df[site_list].values**2 + v_df[site_list].values**2)
    uv_dir_df.loc[:, site_list] = np.arctan2(u_df[site_list].values, v_df[site_list].values)
    uv_dic['uv_speed'] = uv_speed_df
    uv_dic['uv_dir'] = uv_dir_df
    for key, cur_df in uv_dic.items():
        data = cur_df[site_list].values
        u_mean = np.mean(data)
        u_std = np.std(data)
        u_max = np.max(data)
        u_min = np.min(data)

        if key == 'uv_dir':
            u_max = np.pi
            u_min = -np.pi

        data = (data-u_min)/(u_max-u_min)

        u_nmean = np.mean(data)
        u_nstd = np.std(data)
        u_nmax = np.max(data)
        u_nmin = np.min(data)

        cur_df.loc[:, site_list] = data
        stat_df[key] = [u_mean, u_std, u_max, u_min, u_nmean, u_nstd, u_nmax, u_nmin]
        # print(f'key:{key}\n{cur_df}')
        utils.save_hdf2file(cur_df, key, save_path)
    # print(f'stat_df:\n{stat_df}')
    utils.save_hdf2file(stat_df, 'stat', save_path)
def main(args):
    new_list_after_stlplus_2014_2020 = ['1292A', '1203A', '2280A', '2290A', '2997A', '1997A', '1295A', '2315A', '1169A', '1808A', '1226A', '1291A', '2275A', '2298A', '1154A', '2284A', '2271A', '2296A', '1229A', '1170A', '2316A', '2289A', '2007A', '1270A', '1262A', '1159A', '1204A', '2382A', '2285A', '1257A', '1241A', '1797A', '1252A', '1804A', '2342A', '1166A', '1271A', '2360A', '1290A', '1205A', '2312A', '1796A', '1210A', '2299A', '2288A', '2286A', '2314A', '2281A', '1265A', '1242A', '3002A', '1246A', '1167A', '2287A', '2423A', '2282A', '1221A', '2006A', '1171A', '2346A', '2294A', '1799A', '2311A', '3003A', '2001A', '2273A', '2301A', '2383A', '1256A', '2344A', '1145A', '1803A', '1266A', '1147A', '1795A', '2308A', '2357A', '1144A', '1233A', '2000A', '1186A', '2345A', '1294A', '1806A', '1234A', '1298A', '1999A', '2309A', '2278A', '1213A', '2283A', '1264A', '1200A', '1153A', '1240A', '2279A', '2274A', '2306A', '2291A', '1223A', '1239A', '2317A', '2005A', '1212A', '1798A', '1165A', '1215A', '1218A', '2376A', '2379A', '1269A', '1142A', '1228A', '1155A', '2361A', '1149A', '2303A', '2277A', '2310A', '2297A', '2292A', '3004A', '2307A', '1192A', '1267A', '1253A', '2270A', '1196A', '2295A', '1160A', '1235A', '1268A', '1245A', '1794A', '1236A', '1211A', '2004A', '1255A', '1296A', '1232A']

    # nc_data_list = ['cc', 'q', 'crwc', 't', 'u', 'v']
    nc_data_list = ['cc', 'q', 'crwc', 't', 'uv_speed', 'uv_dir']
    pollutant_list = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']

    site_list_save_dir = '/mnt/d/codes/downloads/datasets/国控站点/新站点列表/'
    site_list_save_name = 'csj140.txt'
    nc_dir = '/mnt/d/codes/little programs/国控站点数据处理/datas/netcdf/'
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    site_list_path = site_list_save_dir + site_list_save_name
    lon_index = np.array(list(range(0, 29)))
    lat_index = np.array(list(range(0, 25)))
    
    metro_hdf_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    metro_hdf_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545_sourceAndDiff.h5'
    combine_metro_and_nc_hdf_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    combine_metro_and_nc_hdf_save_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545_sourceAndDiff_nc.h5'
    year_list = range(2016, 2021)

    nc_name = 'guokong176year{}.nc'
    nc2hdf_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    nc2hdf_save_name = f'nc2014_2020长三角{len(new_list_after_stlplus_2014_2020)}.h5'

    # convert_list2txt(new_list_after_stlplus_2014_2020, site_list_save_dir, site_list_save_name)
    # guokong_data_process.see_df(combine_metro_and_nc_hdf_save_dir+combine_metro_and_nc_hdf_save_name)

    # generate_nc2site_hdf(nc_dir, nc_name, lon_index, lat_index, site_path, site_list_path, nc_data_list , nc2hdf_save_dir, nc2hdf_save_name, year_list)
    # guokong_data_process.see_df(combine_metro_and_nc_hdf_save_dir+combine_metro_and_nc_hdf_save_name)
    
    # convert_uv2speed_dir(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, new_list_after_stlplus_2014_2020)

if __name__ == '__main__':
    args = arg_parser()
    main(args)