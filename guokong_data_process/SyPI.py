import pandas as pd
import numpy as np


def SyPI(x, y, x_columns, lags, threshold1, threshold2):
    from rpy2.robjects import numpy2ri
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    numpy2ri.activate()

    '''
    x:[seq_len, in_channels], 
    y:[seq_len, 1], 
    x_columns:[in_channels],
    lags:[in_channels], lag from x to y.
    test_length = seq_len - 2*max_lag_interval
    x_t_i : x[max_lag_interval : **+test_length, i].unsqueeze(1)
    x_t_1_i = x[max_lag_interval-1 : **+test_length, i].unsqueeze(1)
    y_t_wi = y[max_lag_interval+w_i : **+test_length]
    y_t_wi_1 = y[max_lag_interval+w_i-1 : **+test_length]
    s_test += x[max_lag_interval+w_i-w_j-1 : **+test_length].unsqueeze(1)
    '''
    assert(x.shape[1]==len(x_columns)==len(lags))
    causes = []
    condindtest = importr("CondIndTests")

    max_lag_interval = max(int(np.max(lags) - np.min(lags)), 1)
    min_lag_interval = - max_lag_interval
    seq_len = x.shape[0]
    in_channels = x.shape[1]
    test_length = int(seq_len - 2*max_lag_interval)
    lags = [int(lag) for lag in lags]
    # print(f'max_lag_interval:{max_lag_interval}, test_length:{test_length}')
    for i in range(in_channels):
        '''
        生成全遍历完全图
        '''
        w_i = lags[i]
        s_test = np.array([])
        x_t_i = np.expand_dims(x[max_lag_interval : max_lag_interval+test_length, i], axis=1)
        x_t_1_i = np.expand_dims(x[max_lag_interval-1 : max_lag_interval-1+test_length, i], axis=1)
        y_t_wi = y[max_lag_interval+w_i : max_lag_interval+w_i+test_length]
        y_t_wi_1 = y[max_lag_interval+w_i-1 : max_lag_interval+w_i-1+test_length]

        for j in range(in_channels):
            w_j = lags[j]
            if j!=i:
                start_index = max_lag_interval+w_i-w_j-1
                s_test_add = np.expand_dims(x[start_index : start_index+test_length, j], axis=1)
                s_test = np.concatenate((s_test, s_test_add), axis=1) if s_test.size else s_test_add
        
        test1 = condindtest.CondIndTest(x_t_i, y_t_wi, np.concatenate((s_test, y_t_wi_1), axis=1))
        pvalue1 = test1[test1.names.index('pvalue')]
        if pvalue1 < threshold1:
            test2 = condindtest.CondIndTest(x_t_1_i, y_t_wi, np.concatenate((s_test, x_t_i, y_t_wi_1), axis=1))
            pvalue2 = test2[test2.names.index('pvalue')]
            if pvalue2 > threshold2:
                causes.append(x_columns[i])