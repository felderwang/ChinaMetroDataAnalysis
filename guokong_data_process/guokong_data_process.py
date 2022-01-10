import pandas as pd
import numpy as np
import os
import utils

def log_exclude3std_reshape2n01(u):
    '''
    first convert distribution by x := log(x+1), 
    then exclude 3std, 
    finally reshape to [0,1]
    '''

    # u = sr.values
    u_mask = u>=0
    u_0 = u[u>=0]
    u[u<0] = 0.0
    if len(u_0)==0:
        u[np.isnan(u)] = 0.0
        return u, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u_0 = np.log(u_0+1)
    u_mean = np.mean(u_0)
    u_std = np.std(u_0)
    u_max = np.max(u_0)
    # print(f'u_mean:{u_mean} u_std:{u_std}, u_max:{u_max}')

    u_min = min(np.min(u_0), 0.0)
    # print(f'u_mean:{u_mean} u_std:{u_std}, u_max:{u_max}, u_min:{u_min}')

    u_left = u_mean - 3*u_std
    u_right = u_mean + 3*u_std

    u_nega_mask = u_0 < 0
    u_greater_mask = u_0 > u_right
    u_lower_mask = u_0 < u_left

    u_0[u_nega_mask] = np.nan
    u_0[u_greater_mask] = np.nan
    u_0[u_lower_mask] = np.nan

    u[u_mask] = u_0

    u[np.isnan(u)] = 0.0
    u_nmax = np.max(u)
    u_nmin = min(np.min(u), 0.0)

    u = (u-u_nmin)/(u_nmax-u_nmin)
    u_nmean = np.mean(u)
    u_nstd = np.std(u)

    # print(f'u_nmean:{u_nmean}, u_nstd:{u_nstd}, u_nmax:{u_nmax}, u_nmin:{u_nmin}')

    return u, [u_mean, u_std, u_max, u_min, u_nmean, u_nstd, u_nmax, u_nmin]

def deal_guokong(hdf_path, key_list):
    save_path = hdf_path.split('.h5')[0] + 'log_exclude3std_reshape2n01_notstlplus.h5'
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    stat_df = pd.DataFrame()
    stat_name = ['u_mean', 'u_std', 'u_max', 'u_min', 'u_nmean', 'u_nstd', 'u_nmax', 'u_nmin']
    stat_df['stat_name'] = stat_name
    for key in keys:
        if key in key_list:
            print(key)
            df = store.get(key)
            ret_df , stat_list = log_exclude3std_reshape2n01(df.iloc[:, 1:].values)
            if ret_df is not None:
                df.iloc[:, 1:] = ret_df
                
            stat_df[key] = stat_list
            if os.path.exists(save_path):
                df.to_hdf(save_path, key=key, mode='r+', format='table')
                print(f'save to {save_path}, key:{key}, r+')
            else:
                df.to_hdf(save_path, key=key, mode='w', format='table')
                print(f'save to {save_path}, key:{key}, w')
    stat_df.set_index('stat_name', inplace=True)
    stat_df.to_hdf(save_path, key='stat', mode='r+', format='table')
    print(f'save to {save_path}, key:stat, r+')

def divide_hdf2xlsx4rstlplus(hdf_path, save_dir, key_list):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    for key in keys:
        if key in key_list:
            df = store.get(key)
            if key[0] == '/':
                key = key[1:]
            columns = df.columns[1:]  
            for column in columns:
                c_df = df[['datetime', column]]
                c_df[c_df==-9] = np.nan
                # save_name = key + '_' + column + '_nofill.xlsx'
                # c_df.to_excel(save_dir+save_name, index=False)
                save_name = key + '_' + column + '_nofill.csv'
                c_df.to_csv(save_dir+save_name, index=False)

def combine_singleyearh5_2_multiyearh5(hdf_path_list, key_list, save_dir, save_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = save_dir + save_name
    print(f'hdf_path_list:{hdf_path_list}')
    print(f'save_name:{save_name}')
    h5_dic = {}
    for key in key_list:
        h5_dic[key] = []
    for hdf_path in hdf_path_list:
        store = pd.HDFStore(hdf_path)
        keys = store.keys()
        for key in keys:
            if key in key_list:
                df = store.get(key)
                # df[df==0] = np.nan
                h5_dic[key].append(df)
    for key, value in h5_dic.items():
        out_df = pd.concat(value, axis=0, ignore_index=True)
        # print(f'key:{key}, out_df:\n{out_df}')
        if os.path.exists(save_name):
            print(f'save to:{save_name}, key:{key}, r+')
            out_df.to_hdf(save_name, key=key, mode='r+', format='table')
        else:
            print(f'save to:{save_name}, key:{key}, w')
            out_df.to_hdf(save_name, key=key, mode='w', format='table')

def filter_site_list(origin_list, remove_list):
    remove_set = set(remove_list)
    ret_list = []
    for site in origin_list:
        if not site in remove_set:
            ret_list.append(site)
    print(ret_list)

def compose_stplusresult_2_site(file_dir, save_dir, save_name, site_list, cat_list, preffix='stlplus_{}_nofill', suffix='.csv', refer_hdf_path=None, time_index=None, stlplus_parttition_coefficient=[0.02, 0.08, 0.45, 0.45]):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = save_dir + save_name
    print(f'save_name:{save_name}')
    coefficient_sum = 0.0
    for coefficient_num in stlplus_parttition_coefficient:
        coefficient_sum += coefficient_num
    if coefficient_sum != 1:
        assert(0)
    if not refer_hdf_path is None:
        refer_store = pd.HDFStore(refer_hdf_path)
        keys = refer_store.keys()
        key = keys[0]
        refer_df = refer_store.get(key)
        time_index = refer_df['datetime']
    elif time_index is None:
        assert(0)

    for single_cat in cat_list:
        out_df = pd.DataFrame()
        out_df['datetime'] = time_index
        for single_site in site_list:
            if single_cat[0] == '/':
                single_cat = single_cat[1:]
            file_path_bone = file_dir + preffix + f'_{single_cat}_{single_site}' + suffix
            remainder_file_path = file_path_bone.format("remainder")
            seasonal_file_path = file_path_bone.format("seasonal")
            trend_file_path = file_path_bone.format("trend")
            if 'csv' in suffix:
                remainder_df = pd.read_csv(remainder_file_path)
                seasonal_df = pd.read_csv(seasonal_file_path)
                trend_df = pd.read_csv(trend_file_path)
            elif 'xlsx' in suffix:
                remainder_df = pd.read_excel(remainder_file_path)
                seasonal_df = pd.read_excel(seasonal_file_path)
                trend_df = pd.read_excel(trend_file_path)
            remainder_df[remainder_df=='NA'] = np.nan
            remainder_df = remainder_df.astype('float64')

            if len(remainder_df.columns) != len(stlplus_parttition_coefficient):
                assert(0)
            ret_sr = np.zeros(len(seasonal_df))
            for idx in range(len(remainder_df.columns)):
                remainder_sr = remainder_df.iloc[:, idx].values
                seasonal_sr = seasonal_df.iloc[:,idx].values
                trend_sr = trend_df.iloc[:, idx].values
                remainder_sr_0 = remainder_sr[~np.isnan(remainder_sr)]
                remainder_mean = np.mean(remainder_sr_0)
                remainder_std = np.std(remainder_sr_0)
                normal_len = len(remainder_sr) - len(remainder_sr_0)
                remainder_sr[np.isnan(remainder_sr)] = np.random.normal(remainder_mean, remainder_std, normal_len)
                ret_sr += stlplus_parttition_coefficient[idx] * (remainder_sr+seasonal_sr+trend_sr)
            out_df[single_site] = ret_sr
            print(f'single_site:{single_site}, sr_len:{len(remainder_sr)}, normal_len:{normal_len}')

        if os.path.exists(save_name):
            print(f'save to:{save_name}, key:{single_cat}, r+')
            out_df.to_hdf(save_name, key=single_cat, mode='r+', format='table')
        else:
            print(f'save to:{save_name}, key:{single_cat}, w')
            out_df.to_hdf(save_name, key=single_cat, mode='w', format='table')

def add_stats2stlplusedhdf(hdf_path_with_stat, hdf_path_after_stlplus, stat_key='stat'):
    stat_store = pd.HDFStore(hdf_path_with_stat)

    stat_df = stat_store.get(stat_key)
    print(stat_df)
    print(f'save to:{hdf_path_after_stlplus}, key:{stat_key}, r+')
    stat_df.to_hdf(hdf_path_after_stlplus, key=stat_key, mode='r+', format='table')

    stat_store.close()
    after_stlplus_store = pd.HDFStore(hdf_path_after_stlplus)
    after_stlplus_keys = after_stlplus_store.keys()
    print(after_stlplus_keys)
    for key in after_stlplus_keys:
        df = after_stlplus_store.get(key)
        print(df)
    after_stlplus_store.close()
    

def get_diff_hdf(hdf_path, save_dir, save_name, key_list, stat_key='stat'):
    save_name = save_dir + save_name
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    for key in keys:
        if not key in key_list:
            continue
        df = store.get(key)
        columns = df.columns
        diff_df = pd.concat((df[[columns[0]]], df[columns[1:]].diff()),axis=1)
        if key[0] == '/':
            key = key[1:]
        diff_key = key + '_diff'

        if os.path.exists(save_name):
            print(f'save to:{save_name}, key:{key}, {diff_key}, r+')
            df.to_hdf(save_name, key=key, mode='r+', format='table')
        else:
            print(f'save to:{save_name}, key:{key}, {diff_key}, w')
            df.to_hdf(save_name, key=key, mode='w', format='table')

        diff_df.to_hdf(save_name, key=diff_key, mode='r+', format='table')
    stat_df = store.get(stat_key)
    stat_df.to_hdf(save_name, key=stat_key, mode='r+', format='table')

def main():
    origin_list = ["1292A", "1203A", "2280A", "2290A", "1150A", "1230A", "2997A", "1997A", "1295A", "1809A", "3046A", "3195A", "1274A", "3187A", "2315A", "1279A", "1169A", "1808A", "1226A", "1291A", "2907A", "2275A", "2298A", "1154A", "2358A", "2284A", "2271A", "2296A", "1229A", "2873A", "1170A", "2316A", "2289A", "2007A", "3207A", "1270A", "1262A", "2318A", "1159A", "1204A", "2382A", "3190A", "2285A", "1257A", "1241A", "1797A", "1252A", "1804A", "2342A", "1166A", "1271A", "2360A", "1290A", "1205A", "2312A", "1796A", "1800A", "1210A", "2299A", "2343A", "2288A", "2286A", "2314A", "3173A", "2994A", "1146A", "2281A", "1278A", "1265A", "1242A", "3002A", "1246A", "1167A", "2287A", "2423A", "2282A", "1221A", "2006A", "1171A", "2346A", "2294A", "1799A", "2311A", "3003A", "2001A", "2273A", "2301A", "2383A", "1256A", "2344A", "1145A", "1141A", "1803A", "1266A", "1147A", "1795A", "2308A", "2357A", "1263A", "1144A", "1233A", "2000A", "1186A", "3164A", "2345A", "2872A", "1294A", "1806A", "1234A", "1298A", "1999A", "2309A", "2278A", "1213A", "2283A", "1264A", "1200A", "1153A", "1240A", "2279A", "2274A", "2306A", "2291A", "1223A", "3191A", "1239A", "2317A", "2005A", "1212A", "1798A", "1165A", "2875A", "1215A", "1148A", "1218A", "2376A", "2379A", "1269A", "1142A", "1228A", "1155A", "2361A", "2870A", "1149A", "2303A", "2277A", "2310A", "2297A", "2292A", "3004A", "1276A", "2307A", "1192A", "1272A", "1267A", "1253A", "2270A", "1196A", "2295A", "1160A", "1235A", "1268A", "1245A", "1162A", "1794A", "3237A", "1236A", "1801A", "2996A", "1211A", "2921A", "2004A", "1255A", "1227A", "1296A", "1232A"]
    remove_list_2014_2020 = ["3195A", "3187A", "3207A", "3190A", "3173A", "3164A", "3191A", "3237A", "2872A", "2996A", "1230A", "2907A", "2343A", "3173A", "2994A", "1263A", "2870A", "2921A", "1227A", "1150A", "1809A", "3046A", "1274A", "1279A", "2358A", "2873A", "2318A", "1800A", "1146A", "1278A", "1141A", "2875A", "1148A", "1276A", "1272A", "1162A", "1801A"]

    new_list_after_stlplus_2014_2020 = ['1292A', '1203A', '2280A', '2290A', '2997A', '1997A', '1295A', '2315A', '1169A', '1808A', '1226A', '1291A', '2275A', '2298A', '1154A', '2284A', '2271A', '2296A', '1229A', '1170A', '2316A', '2289A', '2007A', '1270A', '1262A', '1159A', '1204A', '2382A', '2285A', '1257A', '1241A', '1797A', '1252A', '1804A', '2342A', '1166A', '1271A', '2360A', '1290A', '1205A', '2312A', '1796A', '1210A', '2299A', '2288A', '2286A', '2314A', '2281A', '1265A', '1242A', '3002A', '1246A', '1167A', '2287A', '2423A', '2282A', '1221A', '2006A', '1171A', '2346A', '2294A', '1799A', '2311A', '3003A', '2001A', '2273A', '2301A', '2383A', '1256A', '2344A', '1145A', '1803A', '1266A', '1147A', '1795A', '2308A', '2357A', '1144A', '1233A', '2000A', '1186A', '2345A', '1294A', '1806A', '1234A', '1298A', '1999A', '2309A', '2278A', '1213A', '2283A', '1264A', '1200A', '1153A', '1240A', '2279A', '2274A', '2306A', '2291A', '1223A', '1239A', '2317A', '2005A', '1212A', '1798A', '1165A', '1215A', '1218A', '2376A', '2379A', '1269A', '1142A', '1228A', '1155A', '2361A', '1149A', '2303A', '2277A', '2310A', '2297A', '2292A', '3004A', '2307A', '1192A', '1267A', '1253A', '2270A', '1196A', '2295A', '1160A', '1235A', '1268A', '1245A', '1794A', '1236A', '1211A', '2004A', '1255A', '1296A', '1232A']
    key_list = ['/CO', '/NO2', '/O3', '/PM10', '/PM2.5', '/SO2']
    nahg_key_list = ['/GEM', '/GOM', '/PBM']
    year_list =range(2014, 2021)
    combine_hdf_path_list = []
    combine_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    combine_save_name = f'guokong{year_list[0]}_{year_list[-1]}长三角176.h5'
    combine_raw_hdf_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2014_2020长三角176log_exclude3std_reshape2n01_notstlplus.h5'
    combine_raw_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/stlplus/2014_2020/raw/'

    stlplus_excel_dir = f'/mnt/d/codes/downloads/datasets/国控站点/stlplus/2014_2020/stlplus/nofill_single/'
    compose_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    compose_save_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545.h5'

    diff_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    diff_save_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545_sourceAndDiff.h5'

    nahg_hdf_path = f'/mnt/d/codes/downloads/datasets/北美汞/h5/AMNet-ALL20092017_p5.h5'
    nahg_divide_save_dir = f'/mnt/d/codes/downloads/datasets/北美汞/stlplus/2009_2017/raw/'



    for year in year_list:
        # year = 2019
        hdf_path = '/mnt/d/codes/downloads/datasets/国控站点/h5/guokong2019长三角mp0.05_len176.h5'
        new_hdf_path = f'/mnt/d/codes/downloads/datasets/国控站点/h5/guokong{year}长三角176.h5'
        raw_hdf_path = f'/mnt/d/codes/downloads/datasets/国控站点/h5/guokong{year}长三角176log_exclude3std_reshape2n01_notstlplus.h5'
        raw_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/stlplus/{year}/raw/'
        combine_hdf_path_list.append(new_hdf_path)
        # utils.see_df(hdf_path)
        # deal_guokong(new_hdf_path, key_list)
        # divide_hdf2xlsx4rstlplus(raw_hdf_path, raw_save_dir, key_list)
    # utils.see_df(nahg_hdf_path)
    # combine_singleyearh5_2_multiyearh5(combine_hdf_path_list, key_list, combine_save_dir, combine_save_name)
    # deal_guokong(combine_save_dir+combine_save_name, key_list)
    # divide_hdf2xlsx4rstlplus(combine_raw_hdf_path, combine_raw_save_dir, key_list)
    # filter_site_list(origin_list, remove_list_2014_2020)
    # compose_stplusresult_2_site(stlplus_excel_dir, compose_save_dir, compose_save_name, new_list_after_stlplus_2014_2020, key_list, refer_hdf_path=combine_raw_hdf_path)

    # divide_hdf2xlsx4rstlplus(nahg_hdf_path, nahg_divide_save_dir, nahg_key_list)

    # add_stats2stlplusedhdf(combine_raw_hdf_path, compose_save_dir+compose_save_name)
    # get_diff_hdf(compose_save_dir+compose_save_name, diff_save_dir, diff_save_name)
    # get_diff_hdf(compose_save_dir+compose_save_name, diff_save_dir, diff_save_name, key_list)
    # utils.see_df(diff_save_dir+diff_save_name)
    
if __name__ == '__main__':
    main()