import numpy as np
import pandas as pd
import pandas_ta as ta

excluded_list = ['Time', 'SQZ_NO', 'SQZPRO_NO']

checked_indicators = {'cycles': ['ebsw'],
                      'momentum': ['ao', 'apo', 'bias', 'bop', 'brar', 'cci', 'cfo', 'cg', 'cmo', 'coppock', 'cti', 'dm', 'er', 'eri', 'fisher', 'inertia', 'kdj', 'kst', 'macd', 'mom', 'pgo', 'ppo', 'pvo', 'roc', 'rsi', 'rsx', 'rvgi', 'slope', 'smi', 'squeeze', 'squeeze_pro', 'stc', 'stoch', 'stochrsi', 'trix', 'tsi', 'uo', 'willr'],
                      'overlap': ['alma', 'dema', 'ema', 'fwma', 'hl2', 'hlc3', 'hma', 'hwma', 'jma', 'kama', 'linreg', 'midpoint', 'ohlc4', 'pwma', 'rma', 'sinwma', 'sma', 'ssf', 'swma', 't3', 'tema', 'trima', 'vidya', 'vwma', 'wcp', 'wma'],
                      'performance': ['log_return', 'percent_return'],
                      'statistics': ['entropy', 'kurtosis', 'mad', 'median', 'quantile', 'skew', 'stdev', 'variance', 'zscore'],
                      'trend': ['adx', 'amat', 'aroon', 'chop', 'cksp', 'decay', 'decreasing', 'dpo', 'increasing', 'long_run', 'qstick', 'short_run', 'tsignals', 'ttm_trend', 'vhf', 'vortex', 'xsignals'],
                      'volatility': ['aberration', 'accbands', 'atr', 'donchian', 'hwc', 'kc', 'massi', 'natr', 'pdist', 'rvi', 'thermo', 'true_range', 'ui'],
                      'volume': ['adosc', 'cmf', 'efi', 'kvo', 'mfi', 'pvol', 'pvr']}

checked_indicators_list = []
for key, value in checked_indicators.items():
    checked_indicators_list.extend(value)
checked_indicators_list = list(np.sort(checked_indicators_list))

adjusted_indicators = {'dpo': ['centered', False]}

check_indicators = {'cycles': [],
                    'momentum': [],
                    'overlap': ['hilo', 'supertrend', 'vwap'],
                    'performance': [],
                    'statistics': [],
                    'trend': ['psar'],
                    'volatility': [],
                    'volume': ['ad', 'aobv', 'nvi', 'obv', 'pvi', 'pvt'],}

exclude_first_n = 700

def test_f1(data_input):
    df = pd.DataFrame(data_input[1][:,[0,1,2,3,4,5,7,8,9,10]],
                      columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df.set_index(pd.DatetimeIndex(df["Time"]), inplace=True)
    df.ta.percent_return(append=True)
    
    for ind in checked_indicators_list:
        test = getattr(df.ta, ind)
        if ind == 'dpo':
            test(centered=False, append=True)
        else:
            test(append=True)
    
    df = df.drop(excluded_list, axis=1)
    # df = df[-200:]
    # df = df[exclude_first_n:]
    # np.save(f'Data/1 hour data/all bybit data/{data_input[0]}.npy', df.values)
    # return data_input[0]
    return data_input[0], df

    
def test_f2(data_input):
    df = pd.DataFrame(data_input[1][:,[0,1,2,3,4,5,7,8,9,10]],
                      columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df.set_index(pd.DatetimeIndex(df["Time"]), inplace=True)
    # df2 = pd.DataFrame(data_input[2][:,[0,1,2,3,4,5,7,8,9,10]],
    #                   columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
    #                              'Taker buy base asset volume', 'Taker buy quote asset volume'])
    # df2.set_index(pd.DatetimeIndex(df2["Time"]), inplace=True)
    df = pd.concat([data_input[2], df], ignore_index=False)
    df.ta.percent_return(append=True)
    
    for ind in checked_indicators_list:
        test = getattr(df.ta, ind)
        if ind == 'dpo':
            test(centered=False, append=True)
        else:
            test(append=True)
            
    df = df.drop(excluded_list, axis=1)
    df = df[-200:]
    # np.save(f'Data/1 hour data/all test trading data/{data_input[0]}.npy', df.values)
    # return data_input[0]
    return data_input[0], df


def calculate_indicators(data_input):
    df = pd.DataFrame(data_input[1][:,[0,1,2,3,4,5,7,8,9,10]],
                      columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df.set_index(pd.DatetimeIndex(df["Time"]), inplace=True)
    df.ta.percent_return(append=True)
    
    for ind in checked_indicators_list:
        test = getattr(df.ta, ind)
        if ind == 'dpo':
            test(centered=False, append=True)
        else:
            test(append=True)
    
    df = df.drop(excluded_list, axis=1)
    # df = df[-200:]
    df = df[exclude_first_n:]
    np.save(f'Data/1 hour data/all new_2 bybit data/{data_input[0]}.npy', df.values)
    return data_input[0]
    # return data_input[0], df