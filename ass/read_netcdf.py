import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import geopandas
import math
import datetime

def draw_uv_field(lon, lat, u_data, v_data, geo_path, site_path, site_txt, save_dir='../pics/uvfield/', save_name='default', draw_indices=False, indices_set=None, tot_minVal=None, tot_maxVal=None):
    color_site = '#0000ff'
    point_size = 15
    color_indice = '#00ff00'
    point_size_indice = 30
    with open(site_txt, 'r') as f:
        lines = f.readlines()
        site_list = [s[:-1] for s in lines]
        # print(site_list)
    df_site = pd.read_excel(site_path)
    df_site = df_site[df_site['监测点编码'].isin(site_list)]

    # print(f'shape: lon:{lon.shape}, lat:{lat.shape}, u_data:{u_data.shape}, v_data:{v_data.shape}')
    lon_new = np.expand_dims(lon, axis=0)
    lon_new = np.repeat(lon_new, len(lat), axis=0)
    lat_new = np.expand_dims(lat, axis=1)
    lat_new = np.repeat(lat_new, len(lon), axis=1)
    # print(f'lon_new:{lon_new.shape}\nlat_new:{lat_new.shape}')
    # print(f'lon_new1:{lon_new[0]}\nlat_new1:{lat_new[0]}')
    
    states = geopandas.read_file(geo_path)

    colors = np.zeros_like(lat_new)
    site_lon = df_site['经度'].values
    site_lat = df_site['纬度'].values
    for lon_index in range(len(lon)):
        for lat_index in range(len(lat)):
            colors[lat_index, lon_index] = (u_data[lat_index, lon_index]**2+v_data[lat_index, lon_index]**2)**0.5
    if tot_minVal is None:
        minVal = colors.min()
        maxVal = colors.max()
    else:
        minVal = tot_minVal
        maxVal = tot_maxVal
        
    fig, axs = plt.subplots(1, 1, figsize=(20,16))
    ax = axs
    ax.title.set_text(f'UV Field, min:{colors.min():.3f}, max:{colors.max():.3f}\ntot min:{minVal:.3f}, max:{maxVal:.3f}')
    im = ax.quiver(lon_new, lat_new, u_data, v_data, colors, cmap='Reds', scale=20, scale_units='xy', clim=(minVal, maxVal))
    fig.colorbar(im)

    if draw_indices:
        for indice in indices_set:
            # print(f'indice:{indice}')
            ax.scatter(lon[indice[0]], lat[indice[1]], color=color_indice, s=point_size_indice)
    ax.scatter(site_lon, site_lat, color=color_site, s=point_size)
    states.boundary.plot(ax=ax)
    ax.set_xlim(lon[0]-0.5, lon[-1]+0.5)
    ax.set_ylim(lat[-1]-0.5, lat[0]+0.5)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    out_name = save_dir+save_name+f'.png'
    print(out_name)
    plt.legend()
    plt.savefig(out_name)
    plt.close()

def read_key(netcdf_dir, lon_index, lat_index, geo_path, site_path, site_txt):
    year_list = list(range(2020, 2021))
    for year in year_list:
        netcdf_path = netcdf_dir + f'guokong176year{year}.nc'
        nc_obj = Dataset(netcdf_path)
        keys = nc_obj.variables.keys()
        
        for level in range(6):
            for key in list(keys):
                print(key)
                # if key == 'longitude' or key == 'latitude':
                    # data = np.array(nc_obj.variables[key][:])
                    # idx = range(0, len(data), 4)
                if key == 'u':
                    data = np.array(nc_obj.variables[key][:])
                    u_field = data[:, level, lat_index[:, None], lon_index]
                elif key == 'v':
                    data = np.array(nc_obj.variables[key][:])
                    v_field = data[:, level, lat_index[:, None], lon_index]
                elif key == 'longitude':
                    data = np.array(nc_obj.variables[key][:])
                    lon = data[lon_index]
                elif key == 'latitude':
                    data = np.array(nc_obj.variables[key][:])
                    lat = data[lat_index]
            tot_windspeed = (u_field**2 + v_field**2) ** 0.5
            tot_minVal = np.min(tot_windspeed)
            tot_maxVal = np.max(tot_windspeed)
            # time_list = np.random.randint(0, len(data), 10)
            # print(f'len data:{len(data)}, time_list:{time_list}')
            # indices_set = get_indices(netcdf_path, lon_index, lat_index, site_path, site_txt)

            # for time in time_list:
            for time in range(len(data)):
                day = time//4
                dt = datetime.datetime(year, 1, 1) + datetime.timedelta(day)
                month = dt.month 
                step = (time - day*4)*6
                day = dt.day
                save_name = f'{str(year)}_{str(month)}_{str(day)}_{str(step)}'
                save_dir = f'../pics/uvfield/{str(year)}_l{str(level)}/'
                draw_uv_field(lon, lat, u_field[time], v_field[time], geo_path, site_path, site_txt, save_dir=save_dir, save_name=save_name, draw_indices=False, tot_minVal=tot_minVal, tot_maxVal=tot_maxVal)

def get_indices(netcdf_path, lon_index, lat_index, site_path, site_txt):
    nc_obj = Dataset(netcdf_path)
    keys = nc_obj.variables.keys()
    indices_set = set()
    for key in list(keys):
        if key == 'longitude':
            data = np.array(nc_obj.variables[key][:])
            lon = data[lon_index]
        elif key == 'latitude':
            data = np.array(nc_obj.variables[key][:])
            lat = data[lat_index]
    with open(site_txt, 'r') as f:
        lines = f.readlines()
        site_list = [s[:-1] for s in lines]
        # print(site_list)
    print(f'lon:{lon}\nlat:{lat}')
    df_site = pd.read_excel(site_path)
    df_site = df_site[df_site['监测点编码'].isin(site_list)]
    for site in site_list:
        site_sr = df_site[df_site['监测点编码']==site]
        site_lon = site_sr['经度'].values[0]
        site_lat = site_sr['纬度'].values[0]
        closest_lon = math.floor(site_lon+0.5)
        closest_lat = math.floor(site_lat+0.5)
        closet_lon_idx = np.where(lon==closest_lon)[0][0]
        closet_lat_idx = np.where(lat==closest_lat)[0][0]
        indices_set.add((closet_lon_idx,closet_lat_idx))
        # print(f'site:{site}\nlon:{site_lon}, lat:{site_lat}\ncloset_lon:{closest_lon}, closest_lat:{closest_lat}\ncloset_lon_idx:{closet_lon_idx}, closet_lat_idx:{closet_lat_idx}')
    # print(f'indices_set:\n{indices_set}')
    return indices_set
def read_key_only(netcdf_dir, lon_index, lat_index, year):
    netcdf_path_list = os.listdir(netcdf_dir)
    print(len(lon_index), len(lat_index))
    for single_netcdf_path in netcdf_path_list:
        if str(year) in single_netcdf_path:
            netcdf_path = netcdf_dir + single_netcdf_path
            nc_obj = Dataset(netcdf_path)
            keys = nc_obj.variables.keys()
            for key in list(keys)[4:]:
                data = np.array(nc_obj.variables[key][:])
                sub_data = data[:, :, lat_index[:, None], lon_index]
                print(f'key:{key}, sub_data:{sub_data.shape}')
def main():
    key_dic = {'longitude': 'longitude', 'latitude': 'latitude', 'level': 'pressure_level', 'time': 'time', 'cc': 'Fraction of cloud cover', 'z': 'Geopotential', 'o3': 'Ozone mass mixing ratio', 'r': 'Relative humidity', 'clwc': 'Specific cloud liquid water content', 'q': 'Specific humidity', 'crwc': 'Specific rain water content', 't': 'Temperature', 'u': 'U component of wind', 'v': 'V component of wind', 'w': 'Vertical velocity'}
    # print(f'lon_indexs:{lon_indexs}, lat_indexs:{lat_indexs}')
    netcdf_path = '../datas/netcdf/guokong176year2010.nc'
    netcdf_dir = '../datas/netcdf/'
    lon_index = np.array(list(range(0, 29)))
    lat_index = np.array(list(range(0, 25)))
    geo_path = '/mnt/d/codes/downloads/datasets/国控站点/cnmap/CHN_adm0.shp'
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    site_txt = '../长三角站点编码mp0.05_len176.txt'

    nahg_netcdf_dir = '../datas/netcdf_nahg/'
    nahg_lon_index = np.array(list(range(0, 81, 4)))
    nahg_lat_index = np.array(list(range(0, 61, 4)))
    # read_key(netcdf_dir, lon_index, lat_index, geo_path, site_path, site_txt)
    # get_indices(netcdf_path, lon_index, lat_index, site_path, site_txt)
    read_key_only(nahg_netcdf_dir, nahg_lon_index, nahg_lat_index, 2020)

if __name__ == '__main__':
    main()