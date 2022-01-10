# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import sys
import numpy as np
import scipy.sparse as sp
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import sys
import os
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import geopandas

def sp_prac():
    np.set_printoptions(precision=2, suppress=True, formatter={'all':lambda x: str(x)})
    max_num = np.iinfo(np.int32).max
    print(max_num)
    graph = np.array([
        [0, 5, 100, max_num, max_num, max_num],
        [max_num, 0, 20, max_num, 3, max_num],
        [max_num, 4, 0, max_num, max_num, max_num],
        [max_num, 2, max_num, 0, 2, max_num],
        [max_num, max_num, max_num, max_num, 0, 10],
        [max_num, max_num, max_num, 3, max_num, 0]
    ])
    full_adj = sp.csgraph.dijkstra(csgraph=graph, directed=True)
    print(full_adj)
    indices = np.array([3, 2, 5])
    adj = sp.csgraph.dijkstra(csgraph=graph, directed=True, indices=indices)
    print(adj)
# This code is contributed by Divyanshu Mehta
def wind_dis(w, base_dis, max_ratio=10.0):
    ret_ratio = 1.0
    if w>=0:
        if w>=4:
            ret_ratio = np.exp(w-4)
        else:
            ret_ratio = 1
    elif w>=- (max_ratio-1):
        ret_ratio = -w+1
    else:
        ret_ratio = max_ratio
    return ret_ratio * base_dis

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
        # closest_lon = math.floor(site_lon+0.5)
        # closest_lat = math.floor(site_lat+0.5)
        # closest_lon_idx = np.where(lon==closest_lon)[0][0]
        # closest_lat_idx = np.where(lat==closest_lat)[0][0]
        closest_lon_idx = (np.abs(lon-site_lon)).argmin()
        closest_lat_idx = (np.abs(lat-site_lat)).argmin()
        indices_set.add((closest_lon_idx,closest_lat_idx))
        # print(f'site:{site}\nlon:{site_lon}, lat:{site_lat}\nclosest_lon_idx:{closest_lon_idx}, closest_lat_idx:{closest_lat_idx}')
    # print(f'indices_set:\n{indices_set}')
    return indices_set

def dis_windfield(u_field, v_field, indices=None):
    #u_field: from west to east, right + 
    #v_field: from south to north, up +
    #graph: [tot_grid, tot_grid]
    #graph[i, j] means Directed Distance from grid[i] to grid[j] 
    #left, right, up, down. left_up, left_down, right_up, right_down
    #row_idx, 小的在北面，大的在南面。小的idx, 是北纬大的；反之大的idx, 是北纬小的

    base_dis = 1.0
    base_diag_dis = 1.4
    max_ratio = 10.0
    tot_grid = u_field.size
    graph = np.empty((tot_grid, tot_grid))
    graph.fill(np.iinfo(np.int32).max)
    uv_field_rows = u_field.shape[0]
    uv_field_cols = u_field.shape[1]
    for row in range(uv_field_rows):
        for col in range(uv_field_cols):
            grid_idx = row*uv_field_cols + col
            #from east to west, left
            if col > 0:
                l_grid_idx = grid_idx - 1
                u_wind = - u_field[row, col]
                graph[grid_idx, l_grid_idx] = wind_dis(u_wind, base_dis)
            #from west to east, right
            if col < uv_field_cols - 1:
                r_grid_idx = grid_idx + 1
                u_wind = u_field[row, col]
                graph[grid_idx, r_grid_idx] = wind_dis(u_wind, base_dis)
            #from south to north, up
            if row > 0:
                u_grid_idx = grid_idx - uv_field_cols
                v_wind = v_field[row, col]
                graph[grid_idx, u_grid_idx] = wind_dis(v_wind, base_dis)
            #from north to south, down
            if row < uv_field_rows - 1:
                d_grid_idx = grid_idx + uv_field_cols
                v_wind = - v_field[row, col]
                graph[grid_idx, d_grid_idx] = wind_dis(v_wind, base_dis)
            #from center to west-north, left-up
            if col > 0 and row > 0:
                lu_grid_idx = grid_idx - 1 - uv_field_cols
                u_wind = u_field[row, col]
                v_wind = v_field[row, col]
                #做射影
                uv_wind = (-u_wind + v_wind)/base_diag_dis
                graph[grid_idx, lu_grid_idx] = wind_dis(uv_wind, base_diag_dis)
            #from center to west-south, left-down
            if col > 0 and row < uv_field_rows -1 :
                ld_grid_idx = grid_idx - 1 + uv_field_cols
                u_wind = u_field[row, col]
                v_wind = v_field[row, col]
                uv_wind = (-u_wind - v_wind)/base_diag_dis
                graph[grid_idx, ld_grid_idx] = wind_dis(uv_wind, base_diag_dis)
            #from center to east-north, right-up
            if col < uv_field_cols - 1 and row > 0:
                ru_grid_idx = grid_idx + 1 - uv_field_cols
                u_wind = u_field[row, col]
                v_wind = v_field[row, col]
                uv_wind = (u_wind + v_wind)/base_diag_dis
                graph[grid_idx, ru_grid_idx] = wind_dis(uv_wind, base_diag_dis)
            #from center to east-south, right-down
            if col < uv_field_cols - 1 and row < uv_field_rows - 1:
                rd_grid_idx = grid_idx + 1 + uv_field_cols
                u_wind = u_field[row, col]
                v_wind = v_field[row, col]
                uv_wind = (u_wind - v_wind)/base_diag_dis
                graph[grid_idx, rd_grid_idx] = wind_dis(uv_wind, base_diag_dis)
    adj = sp.csgraph.dijkstra(csgraph=graph, directed=True, indices=indices)
    # adj = sp.csgraph.floyd_warshall(csgraph=graph, directed=True)
    return adj

def get_wind_adjs(u_fields, v_fields, indices=None):
    #uv_fields:[seq_len, level, lat, lon]
    seq_len = u_fields.shape[0]
    level = u_fields.shape[1]
    lat = u_fields.shape[2]
    lon = u_fields.shape[3]
    if indices is not None:
        len_indices = len(indices)
        ret_adjs = np.zeros((seq_len, level, len_indices, lat*lon))
    else:
        ret_adjs = np.zeros((seq_len, level, lat*lon, lat*lon))
    print(f'u_fields.shape:{u_fields.shape}')
    print(f'seq_len:{seq_len}, level:{level}, lat:{lat}, lon:{lon}')
    for seqlen_idx in range(seq_len):
        print(f'seqlen_idx:{seqlen_idx}')

        for level_idx in range(level):
            sub_u_field = u_fields[seqlen_idx, level_idx]
            sub_v_field = v_fields[seqlen_idx, level_idx]
            ret_adjs[seqlen_idx, level_idx, ...] = dis_windfield(sub_u_field, sub_v_field, indices=indices)

    return ret_adjs

def netcdf_wind_to_dij(netcdf_dir, year_list, site_path, site_txt):
    # year_list = list(range(2020, 2021))
    indices_set = None
    # indices_ar = None
    indices_ar = [635, 241, 256, 506, 308, 658, 560, 303, 79, 455, 372, 248, 152, 32, 400, 304, 650, 526, 345, 221, 75, 486, 719, 612, 216, 599, 7, 279, 368, 255, 148, 270, 642, 13, 520, 337, 341, 398, 332, 704, 223, 595, 456, 275, 497, 634, 281, 253, 664, 37, 74, 531, 435, 78, 311, 119, 613, 191, 491, 247, 108, 519, 249, 153, 125, 414, 701, 675, 181, 192, 342, 346, 222, 126, 387, 359, 448, 363, 267]
    
    # lon_index = np.array(list(range(0, 29, 4)))
    # lat_index = np.array(list(range(0, 25, 4)))
    lon_index = np.array(list(range(0, 29)))
    lat_index = np.array(list(range(0, 25)))

    for year in year_list:
        print(f'year:{year}')
        netcdf_path = netcdf_dir + f'guokong176year{year}.nc'
            
        nc_obj = Dataset(netcdf_path)
        keys = nc_obj.variables.keys()
        
        for key in list(keys):
            # print(key)
            if key == 'u':
                u_data = np.array(nc_obj.variables[key][:])
                u_fields = u_data[:, :, lat_index[:, None], lon_index]
            elif key == 'v':
                data = np.array(nc_obj.variables[key][:])
                v_fields = data[:, :, lat_index[:, None], lon_index]
            elif key == 'longitude':
                data = np.array(nc_obj.variables[key][:])
                lon = data[lon_index]
            elif key == 'latitude':
                data = np.array(nc_obj.variables[key][:])
                lat = data[lat_index]
            if key == 'level':
                l_data = np.array(nc_obj.variables[key][:])
                print(l_data)

        if indices_set is None and indices_ar is None:
            indices_set = get_indices(netcdf_path, lon_index, lat_index, site_path, site_txt)
            indices_list = []
            uv_field_rows = u_fields.shape[2]
            uv_field_cols = u_fields.shape[3]
            for indice in indices_set:
                indice_lon = indice[0]
                indice_lat = indice[1]
                indice_idx = uv_field_cols * indice_lat + indice_lon
                indices_list.append(indice_idx)
            indices_ar = np.array(indices_list) 
        print(f'indices_ar:{indices_ar}\nlen:{len(indices_ar)}')
            # for indice_idx in range(len(indices_ar)):
                # print(indices_ar[indice_idx])
            # assert(0)
        ret_adjs = get_wind_adjs(u_fields, v_fields, indices_ar)
        # ret_adjs = get_wind_adjs(u_fields, v_fields)
        save_dir = '../datas/wind_adj_all/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = save_dir+f'wind_adjs_{year}.npy'
        print(f'file_name:{file_name}, ret_adjs.shape:{ret_adjs.shape}')
        np.save(file_name, ret_adjs)

def main():
    netcdf_dir = '../datas/netcdf/'
    year_list = list(range(2019, 2021))
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    site_txt = '../长三角站点编码mp0.05_len176.txt'
    netcdf_wind_to_dij(netcdf_dir, year_list, site_path, site_txt)
    # sp_prac()

if __name__ == '__main__':
    main()