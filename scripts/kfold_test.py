import os
import pandas as pd
import numpy as np

parent_path = "../save_models/feature_4/multi"

target_list = ['mfp_sum','afp']
target_dict = {'mfp_sum':[], 'afp':[]}

hyper_set = []


for target in target_list:
    print("Evaluation:", target)
    target_path = os.path.join(parent_path, "multi_{}_transfer_Kfold_random".format(target))
    fold = 0
    for dropout in [0]:
        for hidden_size in [200, 300, 400]:
            for ffn_num_layers in [2, 3, 4]:
                fold += 1
                hypers = (dropout, hidden_size, ffn_num_layers)
                hypers = str(hypers)
                hyper_set.append(hypers)
                score_path = os.path.join(os.path.join(target_path, target+"_{}".format(fold)), "quiet.log")
                with open(score_path, 'r') as f:
                    log_file = f.readlines()
                test_rmse_list = []
                for line in log_file:
                    if "==> test rmse" in line:
                        rmse = float(line.split(' ')[6])
                        test_rmse_list.append(rmse)
                
                overall = '{:.2f}'.format(np.average(test_rmse_list)) + ' +/- ' + '{:.2f}'.format(np.std(test_rmse_list))
                target_dict[target].append(overall)
                # print('Fold: ', fold)
                # print("Hyperparameter:", hypers)
                # print("RMSE: {:.5f}".format(rmse))

hyper_set = hyper_set[:9]

record = pd.DataFrame(
    {"Hyperparameter": hyper_set,
    #  "mfp_mean": target_dict["mfp_mean"],
     "mfp_sum": target_dict["mfp_sum"],
     "afp": target_dict["afp"],}
    #  "hybrid_dim0": target_dict["hybrid_dim0"],
    #  "hybrid_dim1": target_dict["hybrid_dim1"]}
)
record.index += 1 
record.to_csv("Kfold_test_random.csv")

# for target in target_list:
#     index_min = np.argmin(target_dict[target])
#     print("="*45)
#     print("For {} model, the best model is:".format(target))
#     print("Fold {}:".format(index_min + 1))
#     print("Hyperparameter: {}".format(hyper_set[index_min]))
#     print("RMSE: {}".format(target_dict[target][index_min]))
#     print("="*45)