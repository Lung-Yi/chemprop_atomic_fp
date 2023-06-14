import pandas as pd
import os
import copy

afp_path = '/home/lungyi/chemprop/save_models/feature_4/multi/multi_afp_transfer'
mfp_sum_path = '/home/lungyi/chemprop/save_models/feature_4/multi/multi_mfp_sum_transfer'

LIST = []
afp_val_record = {'afp_{}'.format(k): [] for k in range(1,10)}
afp_test_record = {'afp_{}'.format(k): [] for k in range(1,10)}
# afp_val_record = dict.fromkeys(['afp_{}'.format(x) for x in [1,2,3,4,5,6,7,8,9]], copy.deepcopy(LIST))
# afp_test_record = dict.fromkeys(['afp_{}'.format(x) for x in [1,2,3,4,5,6,7,8,9]], copy.deepcopy(LIST))

for tune in range(1,11):
    tune_path = os.path.join(afp_path, 'afp_tune_{}'.format(tune))
    for num in [1,2,3,4,5,6,7,8,9]:
        string = 'afp_{}'.format(num)
        log_path = os.path.join(os.path.join(tune_path, string), 'quiet.log')
        with open(log_path, 'r') as f:
            log_file = f.readlines()
        val_rmse = float(log_file[1].split(' ')[6])
        test_rmse = float(log_file[2].split(' ')[5])
        afp_val_record[string].append(val_rmse)
        afp_test_record[string].append(test_rmse)

mfp_sum_val_record = {'mfp_sum_{}'.format(k): [] for k in range(1,10)}
mfp_sum_test_record = {'mfp_sum_{}'.format(k): [] for k in range(1,10)}
# mfp_sum_val_record = dict.fromkeys(['mfp_sum_{}'.format(x) for x in [1,2,3,4,5,6,7,8,9]], copy.deepcopy(LIST))
# mfp_sum_test_record = dict.fromkeys(['mfp_sum_{}'.format(x) for x in [1,2,3,4,5,6,7,8,9]], copy.deepcopy(LIST))

for tune in range(1,11):
    tune_path = os.path.join(mfp_sum_path, 'mfp_sum_tune_{}'.format(tune))
    for num in [1,2,3,4,5,6,7,8,9]:
        string = 'mfp_sum_{}'.format(num)
        log_path = os.path.join(os.path.join(tune_path, string), 'quiet.log')
        with open(log_path, 'r') as f:
            log_file = f.readlines()
        val_rmse = float(log_file[1].split(' ')[6])
        test_rmse = float(log_file[2].split(' ')[5])
        mfp_sum_val_record[string].append(val_rmse)
        mfp_sum_test_record[string].append(test_rmse)

data = pd.DataFrame()
for key, value in afp_val_record.items():
    data[key+'_val'] = value
for key, value in afp_test_record.items():
    data[key+'_test'] = value
for key, value in mfp_sum_val_record.items():
    data[key+'_val'] = value
for key, value in mfp_sum_test_record.items():
    data[key+'_test'] = value

data.to_csv('temp.csv', index=False)