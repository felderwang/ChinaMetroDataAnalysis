import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
# import pykrige.kriging_tools as kt
# from pykrige.ok import OrdinaryKriging

def draw_target_sensors(lats, lons):
    #lat 纬度
    #lon 经度
    color1 = '#005caf'
    points_size = 2

    # lats = np.array(lats)
    # lons = np.array(lons)
    fig, axs = plt.subplots(1,1, figsize=(8,8))
    ax = axs
    ax.title.set_text(f'Target {len(lats)} sensors')
    ax.scatter(lons, lats, color=color1, s=points_size)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')
    plt.savefig('./pics/target_sensor.png')
    plt.close()

def main():
    #lat 纬度
    #lon 经度
    #长三角 28N-34N 115E-122E
    site_list_path = '/mnt/d/codes/downloads/datasets/国控站点/_站点列表/'
    latest_site_path = site_list_path+'站点列表-2021.01.01起.xlsx'
    latest_site = pd.read_excel(latest_site_path)
    # latest_site.replace('-', 0, inplace=True)
    latest_site = latest_site[latest_site['纬度']!='-']
    latest_site = latest_site[latest_site['纬度']>=28.0]
    latest_site = latest_site[latest_site['纬度']<=34.0]
    latest_site = latest_site[latest_site['经度']>=115.0]
    latest_site = latest_site[latest_site['经度']<=122.0]
    # latest_site.fillna(0, inplace=True)
    lons = latest_site['经度'].values
    lats = latest_site['纬度'].values
    latest_site.to_excel('./长三角站点.xlsx')
    draw_target_sensors(lats, lons)
    site_code = latest_site['监测点编码'].values
    print(site_code)
    # site_code.savetxt('./长三角站点编码.txt')
    np.savetxt('./长三角站点编码.txt', site_code,fmt="%s", delimiter=',')
    # site_list_dup = latest_site['监测点编码'].values
    # site_list_dup = ['datetime']+site_list_dup.tolist()#dup 1961A, 1966A
    # site_list = []
    # [site_list.append(x) for x in site_list_dup if x not in site_list]
if __name__=='__main__':
    # main()
    # ar = np.loadtxt('./长三角站点编码.txt')
    # print(ar)
    f = open('./长三角站点编码mp0.1_len228.txt', 'r')
    con = f.read()
    c1 = con.split('\n')
    if c1[-1]=='':
        c1 = c1[:-1]
    print(c1)