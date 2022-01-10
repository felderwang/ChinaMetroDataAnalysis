import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import geopandas
import pandas as pd
import numpy as np
import datetime
import scipy.stats
import statsmodels.api as sm
import os
import math

def draw_site(site_path, geo_path, site_txt):
    color_site = '#ff0000'
    point_size = 10
    font_size = 5
    site_list = []
    site_dic = {}
    l_lon = 1e3
    r_lon = -1e3
    u_lat = -1e3
    d_lat = 1e3
    with open(site_txt, 'r') as f:
        lines = f.readlines()
        site_list = [s[:-1] for s in lines]
        print(site_list)
    df_site = pd.read_excel(site_path)
    df_site = df_site[df_site['监测点编码'].isin(site_list)]
    print(df_site)
    print(f'len site_list:{len(site_list)}, df_site:{len(df_site)}')
    
    # states = geopandas.read_file(geo_path)
    fig, axs = plt.subplots(1, 1, figsize=(20,25))
    ax = axs
    ax.title.set_text(f'China Sites')
    for site in site_list:
        site_sr = df_site[df_site['监测点编码']==site]
        site_lon = site_sr['经度'].values[0]
        site_lat = site_sr['纬度'].values[0]
        if site_lon < l_lon:
            l_lon = site_lon
        if site_lon > r_lon:
            r_lon = site_lon
        if site_lat > u_lat:
            u_lat = site_lat
        if site_lat < d_lat:
            d_lat = site_lat
        # print(f'site:{site}, site_lon:{site_lon}, site_lat:{site_lat}')
        # site_dic[site] = [site_lon, site_lat]
        ax.scatter(site_lon, site_lat, color=color_site, s=point_size)
        ax.annotate(site, (site_lon, site_lat), fontsize=font_size)
        
    # states.boundary.plot(ax=ax)
    print(f'l_lon:{l_lon}, r_lon:{r_lon}, u_lat:{u_lat}, d_lat:{d_lat}')
    plt.savefig('./pics/selected_sites_176_nomap.png')
    plt.close()



def main():
    site_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/站点列表-2021.01.01起.xlsx'
    site_txt = './长三角站点编码mp0.05_len176.txt'
    geo_path = '/mnt/d/codes/downloads/datasets/国控站点/cnmap/CHN_adm0.shp'
    draw_site(site_path, geo_path, site_txt)

if __name__ == '__main__':
    main()