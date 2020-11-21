import os
import pandas as pd
import datetime
import json

# data_path
root_dir = '/home/yueqi/disk2/courses/kdd/Time-Series/'

# 6 columns
# data_paths = [os.path.join(root_dir, 'dataroot/600036_daily.pkl'), os.path.join(root_dir, 'dataroot/600036.XSHG_2020-1-1_2020-9-30_1m.pkl')]
# save_paths = [os.path.join(root_dir, 'data/600036_daily.txt'), os.path.join(root_dir, 'data/600036_minute.txt')]
# reverses = [True, False]
# chosen_attrs = [['open', 'close', 'high', 'low', 'vol', 'amount'], ['open', 'close', 'high', 'low', 'volume', 'money']]
# rename_attrs = [{'vol':'volume', 'amount':'money'}, {}]
# cal_ptc_change_attrs = [['open', 'close', 'high', 'low', 'volume', 'money'], ['open', 'close', 'high', 'low', 'volume', 'money']]
# div_attrs = [{}, {}]
# removes = [False, True]

# 4 columns
# data_paths = [os.path.join(root_dir, 'dataroot/600036_daily.pkl'), os.path.join(root_dir, 'dataroot/600036.XSHG_2020-1-1_2020-9-30_1m.pkl')]
# save_paths = [os.path.join(root_dir, 'data/600036_daily_2.txt'), os.path.join(root_dir, 'data/600036_minute_2.txt')]
# reverses = [True, False]
# chosen_attrs = [['open', 'close', 'high', 'low'], ['open', 'close', 'high', 'low']]
# rename_attrs = [{}, {}]
# cal_ptc_change_attrs = [['open', 'close', 'high', 'low'], ['open', 'close', 'high', 'low']]
# div_attrs = [{}, {}]
# removes = [False, False]

# only minute data
data_paths = [os.path.join(root_dir, 'dataroot/600036.XSHG_2020-1-1_2020-9-30_1m.pkl'), os.path.join(root_dir, 'dataroot/600036.XSHG_2020-1-1_2020-9-30_1m.pkl')]
save_paths = [os.path.join(root_dir, 'data/600036_minute_4.txt'), os.path.join(root_dir, 'data/600036_minute_6.txt')]
reverses = [False, False]
chosen_attrs = [['open', 'close', 'high', 'low'], ['open', 'close', 'high', 'low', 'volume', 'money']]
rename_attrs = [{}, {}]
cal_ptc_change_attrs = [['open', 'close', 'high', 'low'], ['open', 'close', 'high', 'low', 'volume', 'money']]
div_attrs = [{}, {}]
removes = [False, True]

for data_path, save_path, reverse, chosen_attr, rename_attr, cal_ptc_change_attr, div_attr, remove in zip(data_paths, save_paths, reverses, chosen_attrs, rename_attrs, cal_ptc_change_attrs, div_attrs, removes):
    # reverse order to get increasing order
    data = pd.read_pickle(data_path)
    if reverse:
        data = data.iloc[::-1]
    data.reset_index(drop=True, inplace=True)

    # filter out the useful attrs
    data = data[chosen_attr]
    data = data.rename(columns=rename_attr)

    # remove consecutive duplicate row
    if remove:
        # print(data.loc[(data.volume < 30000) | (data.volume < 30000)])
        # data = data.loc[(data.volume > 30000) & (data.volume > 30000)]
        data = data.loc[data.volume != 0]

    # calculate chagne rate
    data[cal_ptc_change_attr] = data[cal_ptc_change_attr].pct_change(periods = 1) + 1
 
    # divide some big values
    for attr, div in div_attr.items():
        data[attr] = data[attr] / div

    # drop the first row and reindex
    data = data.iloc[1:]
    data.reset_index(drop=True, inplace=True)

    # save the data
    data.to_csv(save_path, sep=',', index=False, header=False)



