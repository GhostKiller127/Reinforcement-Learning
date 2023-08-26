import numpy as np
import pandas as pd
import pandas_ta as ta

excluded_list = ['Time', 'DPO_20', 'EOM_14_100000000', 'ICS_26', 'SQZ_NO', 'SQZPRO_NO',
                 'SUPERTl_7_3.0', 'SUPERTs_7_3.0', 'QQEl_14_5_4.236', 'QQEs_14_5_4.236',
                'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 'HILOl_13_21', 'HILOs_13_21',
                'BBL_5_2.0', 'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0']
temp_excluded_list = ['high_Z_30_1', 'low_Z_30_1', 'CTI_12', 'ER_10', 'UI_14', 'VIDYA_14']
excluded_list.extend(temp_excluded_list)

exclude_first_n = 77

def test_f1(data_input):
    df = pd.DataFrame(data_input[1][:,[0,1,2,3,4,5,7,8,9,10]],
                      columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df.set_index(pd.DatetimeIndex(df["Time"]), inplace=True)
    df.ta.percent_return(append=True)
    df.ta.cores = 0
    df.ta.strategy(ta.AllStrategy, timed=False, verbose=False)
    df = df.drop(excluded_list, axis=1)
    # df = df[-200:]
    df = df[exclude_first_n:]
    np.save(f'Data/test data/{data_input[0]}.npy', df.values)
    return data_input[0]
    # return data_input[0], df

def test_f2(data_input):
    df = pd.DataFrame(data_input[1][:,[0,1,2,3,4,5,7,8,9,10]],
                      columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df.set_index(pd.DatetimeIndex(df["Time"]), inplace=True)
    df = pd.concat([data_input[2][-77:], df], ignore_index=False)
    df.ta.percent_return(append=True)
    df.ta.cores = 0
    df.ta.strategy(ta.AllStrategy, timed=False, verbose=False)
    df = df.drop(excluded_list, axis=1)
    df2 = data_input[2].drop(excluded_list, axis=1)
    df = pd.concat([df2[1:], df[-1:]], ignore_index=False)
    return data_input[0], df