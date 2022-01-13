import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dowhy import CausalModel
import numpy as np
import pandas as pd
import graphviz
import networkx as nx 
from cdt.causality.graph import LiNGAM, PC, GES
from cdt.causality.pairwise import ANM

import utils
import os
import SyPI
import argparse
import copent
from tqdm import tqdm
import json

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_pollute', type=utils.str2bool, default=False)
    parser.add_argument('--pollute_idx', type=int, default=0)
    args = parser.parse_args()
    return args

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from gzraphviz library to valid DOT graph format.
    '''
    graph = string.replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph

def causal_discovery(df, labels, save_dir, save_name, function_names=['LiNGAM']):
    total_functions = {
            'LiNGAM' : LiNGAM,
            'PC' : PC,
            'GES' : GES,
        }
    functions = {}
    for function_name in function_names:
        functions[function_name] = total_functions[function_name]
    # functions = {
        # 'LiNGAM' : LiNGAM,
    # }
    for method, lib in functions.items():
        obj = lib()
        output = obj.predict(df)
        adj_matrix = nx.to_numpy_matrix(output)
        adj_matrix = np.asarray(adj_matrix)
        graph_dot = make_graph(adj_matrix, labels)
        local_save_name = save_name.split('.dot')[0]+f'_{method}.dot'
        local_save_path = save_dir + local_save_name
        print(f'Method: {method}, save to:{local_save_path}')

        graph_dot.render(format='png', filename=local_save_path)
    
        # filename = graph.render(format='png', filename=save_dir+save_name)

def pairwise_causal_discovery(df, labels, save_dir, save_name, print_save_flag=True):
    save_path = save_dir + save_name

    obj = ANM()
    try:
        causal_out = obj.predict(df)
        causal_out_dic = dict(zip(labels, causal_out))
    except:
        print(f'pairwise causal ERROR! labels:{labels}')
        causal_out_dic = {}
    if print_save_flag:
        print(f'save to:{save_path}')
    if os.path.exists(save_path):
        with open(save_path, 'r+') as fp:
            file_data = json.load(fp)
            file_data.update(causal_out_dic)
            fp.seek(0)
            json.dump(file_data, fp, indent=4)
        fp.close()
    else:        
        with open(save_path, 'w') as fp:
            json.dump(causal_out_dic, fp, indent=4)
        fp.close()


def discover_with_cdt_dayaverage(hdf_dir, hdf_name, site_list, year_list, save_dir, stat_name='stat'):
    hdf_path = hdf_dir + hdf_name
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    df_dic = {}
    new_keys = []
    for key in keys:
        if 'diff' in key or 'stat' in key:
            continue
        else:
            new_keys.append(key)
    keys = new_keys
    print(f'keys:{keys}')

    stat_df = store.get(stat_name)
    print(f'stat_df:\n{stat_df}')

    for key in keys:
        df = store.get(key)
        df_dic[key] = utils.average_df_by_month(df)
    
    for site in site_list:
        print(f'site:{site}')
        site_df = pd.DataFrame()
        for key in keys:
            df =  df_dic[key]
            ret_df = pd.DataFrame()
            for year in year_list:
                ret_df = pd.concat([ret_df, df.loc[df['datetime'].dt.year==year][site]], axis=0)
            ret_df = utils.restrain_df_by_stat(ret_df, stat_df, key)
            site_df[key] = ret_df.values[:,0]
        print(f'{site_df}')
        labels = [f'{col}' for col in site_df.columns]
        
        save_name = f'{site}_{year_list[0]}_{year_list[-1]}.dot'
        causal_discovery(site_df, labels, save_dir, save_name)

def discover_with_cdt_perhour(hdf_dir, hdf_name, site_list, year_list, save_dir, function_names, freq='h', stat_name='stat'):
    hdf_path = hdf_dir + hdf_name
    store = pd.HDFStore(hdf_path)
    keys = store.keys()
    df_dic = {}
    new_keys = []
    for key in keys:
        if 'diff' in key or 'stat' in key:
            continue
        else:
            new_keys.append(key)
    keys = new_keys
    print(f'keys:{keys}')

    stat_df = store.get(stat_name)
    print(f'stat_df:\n{stat_df}')

    for key in keys:
        df = store.get(key)
        df_dic[key] = df
    
    for year in year_list:
        year_save_dir = save_dir + f'single/{year}/'
        if not os.path.exists(year_save_dir):
            os.makedirs(year_save_dir)
        print(f'year:{year}')
        for site in site_list:
            print(f'site:{site}')
            site_df = pd.DataFrame()
            for key in keys:
                df = df_dic[key]
                ret_df = df.loc[df['datetime'].dt.year==year][[site]]
                ret_df = utils.restrain_df_by_stat(ret_df, stat_df, key)
                site_df[key] = ret_df.values[:,0]
            print(f'site_df:{site_df}')
            labels = [f'{col}' for col in site_df.columns]

            save_name = f'{site}_single{year}.dot'
            causal_discovery(site_df, labels, year_save_dir, save_name)

def discover_graph_causal(hdf_dir, hdf_name, save_dir, pollutant_list, metro_list, site_list, year_list, function_names, freq='d', stat_name='stat', season_list=['spring', 'summer', 'autumn', 'winter'], lags=None, target_lag_name=None):
    # pd.set_option('display.max_columns', 20)
    '''
    lags is a dictionary
    lag_state:
        'single': only have cat shift 0 and shift lag_num,
        'continuous': have cat shift 0 to lag_num, total lag_num+1 columns,
        'no_need':no need for lag
    lags = {
        'PM2.5':[lag_state, lag_num]
        ...
    }

    target_lag_name is a cat. For example, if target_lag_name=='O3', then set lags['O3'] lag_state to 'no_need'
    '''
    print(f'freq:{freq}')
    EPSILON = 1e-8
    hdf_path = hdf_dir + hdf_name
    store = pd.HDFStore(hdf_path)
    df_dic = {}
    cat_keys = pollutant_list + metro_list
    print(f'cat_keys:{cat_keys}')
    stat_df = store.get(stat_name)
    print(f'stat_df:\n{stat_df}')
    for cat_key in cat_keys:
        cat_df = store.get(cat_key)
        cat_df.loc[:, site_list] = utils.restrain_df_by_stat(cat_df[site_list], stat_df, cat_key, metro_list)

        if freq == 'd' or freq == 'D':
            cat_df = utils.average_df_by_month(cat_df)
        df_dic[cat_key] = cat_df
    store.close()

    # for df_key, item in df_dic.items():
        # print(f'df_key:{df_key}\n{item}')
    lag_save_dir = ''

    if lags is not None:
        lag_save_dir = 'lag'
        if target_lag_name is not None:
            target_lag = lags[target_lag_name]
            target_lag[0] = 'no_need'
            lags[target_lag_name] = target_lag
        max_lag = 0
        for _, cat_lag in lags.items():
            lag_save_dir += str(cat_lag[1])
            max_lag = max(max_lag, cat_lag[1])
        lag_save_dir += '/'
    for year in tqdm(year_list):
        year_save_dir = save_dir + lag_save_dir + f'season_uv/{year}/'
        print(f'year_save_dir:{year_save_dir}')
        if not os.path.exists(year_save_dir):
            os.makedirs(year_save_dir)
        print(f'year:{year}')
        for site in tqdm(site_list):
            for season in season_list:
                site_df = pd.DataFrame()
                for cat_key in cat_keys:
                    cat_df = df_dic[cat_key]
                    ret_df = utils.get_season_df(cat_df, season, year)
                    if lags is None:
                        site_df[cat_key] = ret_df[site].values
                    else:
                        cat_lag = lags[cat_key]
                        single_site_df = ret_df[site].values
                        s0_df = single_site_df[max_lag:]
                        s0_name = cat_key + 's0'
                        site_df[s0_name] = s0_df

                        lag_num = cat_lag[1]
                        if cat_lag[0] == 'no_need':
                            pass
                        elif cat_lag[0] == 'single':
                            if lag_num != 0:
                                slag_name = cat_key + f's{lag_num}'
                                slag_df = single_site_df[max_lag-lag_num:-lag_num]
                                site_df[slag_name] = slag_df

                        elif cat_lag[0] == 'continuous':
                            for lag_idx in range(1, lag_num + 1):
                                slag_name = cat_key + f's{lag_idx}'
                                slag_df = single_site_df[max_lag-lag_idx:-lag_idx]
                                site_df[slag_name] = slag_df
                # print(f'site_df:\n{site_df}')
                site_df[site_df<EPSILON] = 0.0
                labels = [f'{col}' for col in site_df.columns]
                for function_name in function_names:
                    function_year_save_dir = year_save_dir + f'{function_name}/'
                    save_name = f'{site}_single{year}_{season}.dot'
                    try:
                        causal_discovery(site_df, labels, function_year_save_dir, save_name, [function_name])
                    except:
                        print("R ERROR!")

def compute_transfer_entropy_single_site_different_cats(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, save_dir, save_name, site_list, year_list, pollutant_list, metro_list, stat_name = 'stat', args=None, freq='h', lag_range=range(1,9), season_list=None):
    def generate_causal_df_perkey_peryear(first_lag_index, first_df, second_df, freq, lag_range):
        causal_df = pd.DataFrame()
        causal_df['index'] = first_lag_index
        for single_site in tqdm(site_list):
        # for single_site in site_list:
            first_ar = first_df[single_site].values
            second_ar = second_df[single_site].values            
            if freq == 'h' or freq == 'H':
                hour_lag = []
                for lag in lag_range:
                    hour_lag.append(copent.transent(second_ar, first_ar, lag))
                causal_df[single_site] = np.array(hour_lag)
            elif freq == 'd' or freq == 'D':
                day_lag = []
                for lag in lag_range:
                    day_lag.append(copent.transent(second_ar, first_ar, lag))
                causal_df[single_site] = np.array(day_lag)
        causal_df.set_index('index', inplace=True)
        print(f'casual_df:\n{causal_df}')
        return causal_df

    #keys: cat2cat
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    combine_path = combine_metro_and_nc_hdf_save_dir + combine_metro_and_nc_hdf_save_name
    save_path = save_dir + save_name
    combine_store = pd.HDFStore(combine_path)

    stat_df = combine_store.get(stat_name)
    df_dic = {}
    print(f'pollutant_list:{pollutant_list}, metro_list:{metro_list}')
    for combine_key in pollutant_list + metro_list:
        if combine_key[0] == '/':
            combine_key = combine_key[1:]
        combine_df = combine_store.get(combine_key)
        combine_df.loc[:, site_list] = utils.restrain_df_by_stat(combine_df[site_list], stat_df, combine_key)
        df_dic[combine_key] = combine_df

    combine_store.close()
    print(f'df_dic:\n{df_dic}')
    print(f'data pre processing DONE!')
    print(f'freq:{freq}, lag_range:{lag_range}')

    if season_list is not None:
        season_saved_keys_dic = {}
        for season in season_list:
            season_saved_keys_dic[season] = []
            season_save_path = save_dir + save_name.split('.h5')[0]+f'_{season}.h5'
            if os.path.exists(season_save_path):
                season_store = pd.HDFStore(season_save_path)
                season_keys = season_store.keys()
                season_store.close()
                new_season_keys = []
                for season_key in season_keys:
                    if season_key[0] == '/':
                        season_key = season_key[1:]
                    new_season_keys.append(season_key)
                season_saved_keys_dic[season] = new_season_keys

    for first_key in pollutant_list:
    # for first_key in metro_list:
        for second_key in pollutant_list:
            if args.single_pollute:
                second_key = pollutant_list[args.pollute_idx]
                print(f'********pollutant select ON********')
                print(f'pollute_idx:{args.pollute_idx}')
            else:
                print(f'********pollutant select OFF********')

            causal_key = first_key + "2" + second_key
            print(f'causal_key:{causal_key}')

            causal_out_df = pd.DataFrame()
            if season_list is not None:
                have_saved_keys = 0
                season_causal_out_df_dic = {}
                for season in season_list:
                    season_keys = season_saved_keys_dic[season]
                    if causal_key in season_keys:
                        have_saved_keys += 1
                    
                    season_causal_out_df_dic[season] = pd.DataFrame()
                if have_saved_keys == len(season_list):
                    print(f'causal key:{causal_key} have been saved, continue.')
                    continue
            
            
            total_first_df = df_dic[first_key]
            total_second_df = df_dic[second_key]
            if freq == 'd' or freq == 'D':
                total_first_df = utils.average_df_by_month(total_first_df)
                total_second_df = utils.average_df_by_month(total_second_df)
            for year in tqdm(year_list):
                print(f'year:{year}')
                first_lag_index = []
                for lag in lag_range:
                    if freq == 'h' or freq == 'H':
                        first_lag_index.append(f"{year}hour_lag"+str(lag))
                    elif freq == 'd' or freq == 'D':
                        first_lag_index.append(f"{year}day_lag"+str(lag))
                        
                if season_list is not None:
                    for season in season_list:
                        if freq == 'h' or freq == 'H':
                            first_df_generator = utils.get_season_df_per_month_generator(total_first_df, season, year)
                            second_df_generator = utils.get_season_df_per_month_generator(total_second_df, season, year)
                            for i, month_first_df in enumerate(first_df_generator):
                                month_second_df = second_df_generator.__next__()
                                print(f'freq hour, df len:{len(month_first_df.index)}, {len(month_second_df.index)}')
                                # print(f'first df:\n{month_first_df}\nsecond df:\n{month_second_df}')

                                month_first_lag_index = [single_lag_index+f'_m{i}' for single_lag_index in first_lag_index]
                                season_causal_ret_df = generate_causal_df_perkey_peryear(month_first_lag_index, month_first_df, month_second_df, freq=freq, lag_range=lag_range)
                                season_causal_out_df = season_causal_out_df_dic[season]
                                season_causal_out_df = pd.concat([season_causal_out_df, season_causal_ret_df], axis=0)
                                season_causal_out_df_dic[season] = season_causal_out_df
                        elif freq == 'd' or freq == 'D':
                            first_df = utils.get_season_df(total_first_df, season, year)
                            second_df = utils.get_season_df(total_second_df, season, year)
                            season_causal_ret_df = generate_causal_df_perkey_peryear(first_lag_index, first_df, second_df, freq=freq, lag_range=lag_range)
                            season_causal_out_df = season_causal_out_df_dic[season]
                            season_causal_out_df = pd.concat([season_causal_out_df, season_causal_ret_df], axis=0)
                            season_causal_out_df_dic[season] = season_causal_out_df
                else:
                    first_df = total_first_df.loc[total_first_df['datetime'].dt.year==year]
                    second_df = total_second_df.loc[total_second_df['datetime'].dt.year==year]
                    print(f'first_df:\n{first_df}\nsecond_df:\n{second_df}')
                    causal_ret_df = generate_causal_df_perkey_peryear(first_lag_index,  first_df, second_df, freq=freq, lag_range=lag_range)
                    causal_out_df = pd.concat([causal_out_df, causal_ret_df], axis=0)
            if season_list is not None:
                for season, season_out_df in season_causal_out_df_dic.items():
                    season_save_path = save_dir + save_name.split('.h5')[0]+f'_{season}.h5'
                    utils.save_hdf2file(season_out_df, causal_key, season_save_path)
            else:
                utils.save_hdf2file(causal_out_df, causal_key, save_path)

            if args.single_pollute:
                break

def discover_sypi(hdf_dir, hdf_name, save_dir, save_name, site_list, pollutant_list, metro_list, lags, year_list, threshold1=0.01, threshold2=0.2, stat_name='stat', freq='h', season_list=None):
    def generate_sypi_peryear(site_df, target_column, total_columns, lags, threshold1, threshold2):
        total_columns.remove(target_column)
        cause_columns = list(total_columns)
        y = site_df[[target_column]].values
        x = site_df[cause_columns].values
        if lags is None:
            lags = np.ones(x.shape[1])
        return SyPI.SyPI(x, y, cause_columns, lags, threshold1, threshold2)
    print(f'threshold1:{threshold1} threshold2:{threshold2}')
    hdf_path = hdf_dir + hdf_name
    save_path = save_dir + save_name
    print(f'save_path:{save_path}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    store = pd.HDFStore(hdf_path)
    cat_keys = store.keys()
    site_df_dic = {}
    site_cause_dic = {}
    time_flag = False
    stat_df = store.get(stat_name)
    total_columns = set()
    for cat_key in cat_keys:
        if cat_key[0] == '/':
            cat_key = cat_key[1:]
        if (not cat_key in metro_list) and (not cat_key in pollutant_list):
            continue
        total_columns.add(cat_key)

        cat_df = store.get(cat_key)
        cat_df.loc[:, site_list] = utils.restrain_df_by_stat(cat_df[site_list], stat_df, cat_key)
        if freq == 'd' or freq == 'D':
            cat_df = utils.average_df_by_month(cat_df)
        # print(f'cat_key:{cat_key}\n{cat_df}')
        if not time_flag:
            time_index = cat_df['datetime']
            site_df = pd.DataFrame()
            site_df['datetime'] = time_index
            for site in site_list:
                site_df_dic[site] = site_df.copy()
                site_cause_dic[site] = []
            time_flag = True
        for site in site_list:
            site_df = site_df_dic[site]
            site_df[cat_key] = cat_df[site]
            site_df_dic[site] = site_df
    store.close()

    print(f'total_columns:{total_columns}')
    target_columns = ['O3', 'PM2.5']

    for site_key, site_df in tqdm(site_df_dic.items()):
        print(f'site_key:{site_key}')
        target_causes_dic = {}
        for target_column in target_columns:
            for year in year_list:
                target_causes_year_dic = {}
                if season_list is not None:
                    target_causes_year_season_dic = {}
                    for season in season_list:
                        if freq == 'h' or freq == 'H':
                            target_causes_year_season_month_dic = {}
                            site_season_df_generator = utils.get_season_df_per_month_generator(site_df, season, year)
                            for i, month_site_season_df in enumerate(site_season_df_generator):
                                tot_columns = total_columns.copy()
                                target_causes_year_season_month_dic[i] = generate_sypi_peryear(month_site_season_df, target_column, tot_columns, lags, threshold1, threshold2)
                            target_causes_year_season_dic[season] = target_causes_year_season_month_dic

                        elif freq == 'd' or freq == 'D':
                            site_season_df = utils.get_season_df(site_df, season, year)
                            tot_columns = total_columns.copy()
                            target_causes_year_season_dic[season] = generate_sypi_peryear(site_season_df, target_column, tot_columns, lags, threshold1, threshold2)
                    target_causes_year_dic[year] = target_causes_year_season_dic

                else:
                    site_year_df = site_df.loc[site_df['datetime'].dt.year==year]
                    tot_columns = total_columns.copy()
                    target_causes_year_dic[year] = generate_sypi_peryear(site_year_df, target_column, tot_columns, lags, threshold1, threshold2)

            target_causes_dic[target_column] = target_causes_year_dic
            site_cause_dic[site_key] = target_causes_dic

    with open(save_path, 'w') as fp:
        json.dump(site_cause_dic, fp)

def discover_pairwise_anm(hdf_dir, hdf_name, save_dir, save_name, site_list, pollutant_list, metro_list, year_list, stat_name='stat', freq='h', season_list=None):
    print(f'pollutant_list:{pollutant_list}')
    print(f'freq:{freq}')
    df_dic = {}
    site_pairwise_df_dic = {}
    hdf_path = hdf_dir + hdf_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    store = pd.HDFStore(hdf_path)
    cat_keys = store.keys()
    stat_df = store.get(stat_name)
    for cat_key in cat_keys:
        if cat_key[0] == '/':
            cat_key = cat_key[1:]
        if (not cat_key in metro_list) and (not cat_key in pollutant_list):
            continue
        cat_df = store.get(cat_key)
        cat_df.loc[:, site_list] = utils.restrain_df_by_stat(cat_df[site_list], stat_df, cat_key)

        if freq == 'd' or freq == 'D':
            cat_df = utils.average_df_by_month(cat_df)
        df_dic[cat_key] = cat_df
    store.close()
    discover_pairwise_df = pd.DataFrame(columns=['cause', 'effect'])
    #cause -> effect
    for site in site_list:
        site_pairwise_df_dic[site] = discover_pairwise_df.copy()
        
    # for first_key in pollutant_list:
    for first_key in pollutant_list+ metro_list:
        for second_key in pollutant_list:
            if args.single_pollute:
                second_key = pollutant_list[args.pollute_idx]
                print(f'********pollutant select ON********')
                print(f'pollute_idx:{args.pollute_idx}')
            else:
                print(f'********pollutant select OFF********')
            causal_key = first_key + "2" + second_key
            print(f'causal_key:{causal_key}')

            total_first_df = df_dic[first_key]
            total_second_df = df_dic[second_key]
            for year in year_list:
                if season_list is not None:
                    for season in season_list:
                        if freq == 'h' or freq == 'H':
                            pass
                        elif freq == 'd' or freq == 'D':
                            pass
                        season_year_first_df = utils.get_season_df(total_first_df, season, year)
                        season_year_second_df = utils.get_season_df(total_second_df, season, year)
                        season_year_causal_key = causal_key+f'_{year}_{season}'
                        for site in site_list:
                            site_df = site_pairwise_df_dic[site]
                            append_df = pd.DataFrame({'cause':[season_year_first_df[site].values], 'effect':[season_year_second_df[site].values]}, index=[season_year_causal_key])
                            site_df = pd.concat([site_df, append_df], axis=0)
                            site_pairwise_df_dic[site] = site_df
            if args.single_pollute:
                break
    for site, site_df in tqdm(site_pairwise_df_dic.items()):
        # print(f'site:{site}, site_df:\n{site_df}')
        site_columns = site_df.index
        site_save_name = save_name.split('.json')[0] + f'_{site}.json'
        for i, site_column in enumerate(site_columns):
            sub_label = [site_column]
            sub_site_df = site_df.loc[sub_label, :]
            # print(f'sub_site_df:{sub_site_df}\n, sub_label:{sub_label}')
            pairwise_causal_discovery(sub_site_df, sub_label, save_dir, site_save_name, print_save_flag=not bool(i))
        # pairwise_causal_discovery(site_df, site_columns, save_dir, site_save_name)


def main(args):
    new_list_after_stlplus_2014_2020 = ['1292A', '1203A', '2280A', '2290A', '2997A', '1997A', '1295A', '2315A', '1169A', '1808A', '1226A', '1291A', '2275A', '2298A', '1154A', '2284A', '2271A', '2296A', '1229A', '1170A', '2316A', '2289A', '2007A', '1270A', '1262A', '1159A', '1204A', '2382A', '2285A', '1257A', '1241A', '1797A', '1252A', '1804A', '2342A', '1166A', '1271A', '2360A', '1290A', '1205A', '2312A', '1796A', '1210A', '2299A', '2288A', '2286A', '2314A', '2281A', '1265A', '1242A', '3002A', '1246A', '1167A', '2287A', '2423A', '2282A', '1221A', '2006A', '1171A', '2346A', '2294A', '1799A', '2311A', '3003A', '2001A', '2273A', '2301A', '2383A', '1256A', '2344A', '1145A', '1803A', '1266A', '1147A', '1795A', '2308A', '2357A', '1144A', '1233A', '2000A', '1186A', '2345A', '1294A', '1806A', '1234A', '1298A', '1999A', '2309A', '2278A', '1213A', '2283A', '1264A', '1200A', '1153A', '1240A', '2279A', '2274A', '2306A', '2291A', '1223A', '1239A', '2317A', '2005A', '1212A', '1798A', '1165A', '1215A', '1218A', '2376A', '2379A', '1269A', '1142A', '1228A', '1155A', '2361A', '1149A', '2303A', '2277A', '2310A', '2297A', '2292A', '3004A', '2307A', '1192A', '1267A', '1253A', '2270A', '1196A', '2295A', '1160A', '1235A', '1268A', '1245A', '1794A', '1236A', '1211A', '2004A', '1255A', '1296A', '1232A']
    metro_pollutant_hdf_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    metro_pollutant_hdf_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545_sourceAndDiff_nc.h5'
    year_list = range(2016, 2021)
    # nc_data_list = ['cc', 'q', 'crwc', 't', 'uv_speed', 'uv_dir']
    nc_data_list = ['cc', 'q', 't', 'uv_speed', 'uv_dir']
    pollutant_list = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    causal_discovery_season_year_list = range(2017, 2021)
    combine_metro_and_nc_hdf_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    combine_metro_and_nc_hdf_save_name = f'guokong2014_2020长三角140log_exclude3std_reshape2n01_withstlplus284545_sourceAndDiff_nc.h5'
    season_year_list = range(2018, 2020)
    season_list = ['summer', 'winter']

    causal_function_names = ['LiNGAM', 'PC', 'GES']
    # causal_function_names = ['PC', 'GES']
    causal_result_save_dir = f'../pics/causal_result/'
    causal_result_day_save_dir = f'../pics/causal_result/day/'
    causal_result_hour_save_dir = f'../pics/causal_result/hour/'
    causal_day_lags = {
        'cc':['single', 2],
        'q':['single', 2],
        't':['continuous', 4],
        'uv_speed':['continuous', 2],
        'uv_dir':['no_need', 4],
        'CO':['continuous', 2],
        'NO2':['continuous', 2],
        'O3':['continuous', 2],
        'PM10':['continuous', 2],
        'PM2.5':['continuous', 2],
        'SO2':['single', 2],
    }
    
    causal_metro_list = ['cc', 'q', 't', 'uv_speed', 'uv_dir']
    causal_day_freq = 'd'
    causal_hour_freq = 'h'

    transfer_hdf_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    transfer_hdf_name = f'transfer_entropy2014_2020长三角{len(new_list_after_stlplus_2014_2020)}.h5'
    transfer_hdf_day_name = f'transfer_entropy2016_2020长三角{len(new_list_after_stlplus_2014_2020)}_day.h5'
    transfer_hdf_hour_name = f'transfer_entropy2016_2020长三角{len(new_list_after_stlplus_2014_2020)}_hour.h5'
    transfer_day_lag_range = range(1,8)
    transfer_hour_lag_range = range(0,24,4)
    transfer_day_freq = 'd'
    transfer_hour_freq = 'h'

    # sypi_hdf_dir = f'/mnt/d/codes/downloads/datasets/国控站点/h5/'
    # sypi_hdf_name = f'sypi.h5'
    sypi_json_dir = f'/mnt/d/codes/downloads/datasets/国控站点/jsons/'
    sypi_json_name = f'sypi.json'
    sypi_year_list = range(2018, 2021)
    sypi_threshold1 = 0.1
    sypi_threshold2 = 0.2
    sypi_day_freq = 'd'
    sypi_hour_freq = 'h'
    sypi_season_year_list = range(2018, 2020)
    sypi_season_list = ['summer', 'winter']
    sypi_json_season_day_name = f'sypi_season_day.json'
    sypi_json_season_hour_name = f'sypi_season_hour.json'

    pairwise_anm_day_freq = 'd'
    pairwise_anm_hour_freq = 'h'
    pairwise_anm_season_year_list = range(2017, 2020)
    pairwise_anm_season_list = ['spring', 'summer','autumn',  'winter']
    pairwise_anm_season_hour_list = ['spring', 'summer','autumn',  'winter']
    pairwise_anm_day_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/jsons/pairwise_anm/day/'
    pairwise_anm_hour_save_dir = f'/mnt/d/codes/downloads/datasets/国控站点/jsons/pairwise_anm/hour/'
    pairwise_anm_json_season_day_name = f'pairwise_anm_season_day.json'
    pairwise_anm_json_season_hour_name = f'pairwise_anm_season_hour.json'
    

    # discover_with_cdt_dayaverage(metro_pollutant_hdf_dir, metro_pollutant_hdf_name, new_list_after_stlplus_2014_2020, year_list, causal_result_save_dir)


    #discover graph causal per year per day by season
    # discover_graph_causal(metro_pollutant_hdf_dir, metro_pollutant_hdf_name, causal_result_day_save_dir, pollutant_list, causal_metro_list, new_list_after_stlplus_2014_2020, causal_discovery_season_year_list, causal_function_names, freq=causal_day_freq)

    #discover graph causal per year per day by season with lags
    discover_graph_causal(metro_pollutant_hdf_dir, metro_pollutant_hdf_name, causal_result_day_save_dir, pollutant_list, causal_metro_list, new_list_after_stlplus_2014_2020, causal_discovery_season_year_list, causal_function_names, freq=causal_day_freq, lags=causal_day_lags, target_lag_name='O3')

    #discover graph causal per year per hour by season
    # discover_graph_causal(metro_pollutant_hdf_dir, metro_pollutant_hdf_name, causal_result_hour_save_dir, pollutant_list, causal_metro_list, new_list_after_stlplus_2014_2020, causal_discovery_season_year_list, causal_function_names, freq=causal_hour_freq)

    # discover_with_cdt_perhour(metro_pollutant_hdf_dir, metro_pollutant_hdf_name, new_list_after_stlplus_2014_2020, year_list, causal_result_save_dir)

    # combine_metro_and_nc_hdf(metro_hdf_dir, metro_hdf_name, nc2hdf_save_dir, nc2hdf_save_name, combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name)

    # compute_transfer_entropy_single_site_different_cats(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, transfer_hdf_dir, transfer_hdf_name, new_list_after_stlplus_2014_2020, year_list, pollutant_list=pollutant_list, metro_list=nc_data_list, args=args)

    # compute_transfer_entropy_single_site_different_cats(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, transfer_hdf_dir, transfer_hdf_day_name, new_list_after_stlplus_2014_2020, year_list, pollutant_list=pollutant_list, metro_list=nc_data_list, args=args, freq=transfer_day_freq, lag_range=transfer_day_lag_range)

    #compute transfer entropy per year per day, summer and winter
    # compute_transfer_entropy_single_site_different_cats(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, transfer_hdf_dir, transfer_hdf_day_name, new_list_after_stlplus_2014_2020, season_year_list, pollutant_list=pollutant_list, metro_list=nc_data_list, args=args, freq=transfer_day_freq, lag_range=transfer_day_lag_range, season_list=season_list)

    # compute transfer entropy per year per month per hour, summer and winter
    # compute_transfer_entropy_single_site_different_cats(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, transfer_hdf_dir, transfer_hdf_hour_name, new_list_after_stlplus_2014_2020, season_year_list, pollutant_list=pollutant_list, metro_list=nc_data_list, args=args, freq=transfer_hour_freq, lag_range=transfer_hour_lag_range, season_list=season_list)

    #discover sypi per year per day
    # discover_sypi(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, sypi_json_dir, sypi_json_name, new_list_after_stlplus_2014_2020, pollutant_list, nc_data_list, lags=None, year_list=sypi_year_list, threshold1=sypi_threshold1, threshold2=sypi_threshold2, freq=sypi_day_freq)

    #discover sypi, per year per day by season
    # discover_sypi(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, sypi_json_dir, sypi_json_season_day_name, new_list_after_stlplus_2014_2020, pollutant_list, nc_data_list, lags=None, year_list=sypi_season_year_list, threshold1=sypi_threshold1, threshold2=sypi_threshold2, freq=sypi_day_freq, season_list=sypi_season_list)

    #discover sypi, per year per hour per month by season
    # discover_sypi(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, sypi_json_dir, sypi_json_season_hour_name, new_list_after_stlplus_2014_2020, pollutant_list, nc_data_list, lags=None, year_list=sypi_season_year_list, threshold1=sypi_threshold1, threshold2=sypi_threshold2, freq=sypi_hour_freq, season_list=sypi_season_list)

    #discover pairwise, per year per day by season
    # discover_pairwise_anm(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, pairwise_anm_day_save_dir, pairwise_anm_json_season_day_name, new_list_after_stlplus_2014_2020, pollutant_list, causal_metro_list, pairwise_anm_season_year_list, freq=pairwise_anm_day_freq, season_list=pairwise_anm_season_list)

    # discover pairwise, per year per hour by season
    # discover_pairwise_anm(combine_metro_and_nc_hdf_save_dir, combine_metro_and_nc_hdf_save_name, pairwise_anm_hour_save_dir, pairwise_anm_json_season_hour_name, new_list_after_stlplus_2014_2020, pollutant_list, causal_metro_list, pairwise_anm_season_year_list, freq=pairwise_anm_hour_freq, season_list=pairwise_anm_season_hour_list)
    # SyPI.foo()

if __name__ == '__main__':
    args = arg_parser()
    main(args)