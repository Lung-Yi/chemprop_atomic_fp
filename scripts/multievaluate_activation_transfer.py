import os
import pandas as pd
import numpy as np

parent_path = "../save_models/feature_4/multi"

target_list = ['mfp_mean', 'mfp_sum','afp', 'hybrid_dim0', 'hybrid_dim1']
# target_dict = {'mfp_mean':[], 'mfp_sum':[], 'afp':[],'hybrid_dim0':[], 'hybrid_dim1':[]}
# target_list = ['mfp_sum','afp']
target_dict_val_test = {'mfp_mean':[], 'mfp_sum':[], 'afp':[],'hybrid_dim0':[], 'hybrid_dim1':[]}
target_dict_val = {'mfp_mean':[], 'mfp_sum':[], 'afp':[],'hybrid_dim0':[], 'hybrid_dim1':[]}

hyper_set = []

for target in target_list:
    print("Evaluation:", target)
    
    
    for activation in ['ReLU', 'PReLU', 'tanh']:
        fold = 0
        target_path = os.path.join(parent_path, "multi_{}_{}_transfer_newLR_gpu".format(target, activation))
        for hidden_size in [200, 300, 400]:
            for ffn_num_layers in [2, 3, 4]:
                fold += 1
                hypers = (activation, hidden_size, ffn_num_layers)
                hypers = str(hypers)
                hyper_set.append(hypers)
                score_path = os.path.join(os.path.join(target_path, target+"_{}".format(fold)), "test_scores.csv")
                df = pd.read_csv(score_path)
                test_rmse = float(df['Mean rmse'].values[0])
                
                log_path = os.path.join(os.path.join(target_path, target+"_{}".format(fold)), "quiet.log")
                with open(log_path, 'r') as g:
                    log = g.readlines()
                val_rmse = float(log[1].split(' ')[6])
                target_dict_val[target].append(val_rmse)
                target_dict_val_test[target].append('{:.2f} / {:.2f}'.format(val_rmse, test_rmse))
                # print('Fold: ', fold)
                # print("Hyperparameter:", hypers)
                # print("RMSE: {:.5f}".format(rmse))

hyper_set = hyper_set[:27]

# record = pd.DataFrame(
#     {"Hyperparameter": hyper_set,
#      "mfp_mean": target_dict["mfp_mean"],
#      "mfp_sum": target_dict["mfp_sum"],
#      "afp": target_dict["afp"],
#      "hybrid_dim0": target_dict["hybrid_dim0"],
#      "hybrid_dim1": target_dict["hybrid_dim1"]}
# )
record = pd.DataFrame(
    {"Hyperparameter": hyper_set,
     "mfp_sum": target_dict_val_test["mfp_sum"],
     "mfp_mean": target_dict_val_test["mfp_mean"],
     "afp": target_dict_val_test["afp"],
     "hybrid_dim0": target_dict_val_test["hybrid_dim0"],
     "hybrid_dim1": target_dict_val_test["hybrid_dim1"]}
)
record.index += 1
record.to_csv("record_transfer_activation_val_test.csv")

record_val = pd.DataFrame(
    {"Hyperparameter": hyper_set,
     "mfp_sum": target_dict_val["mfp_sum"],
     "mfp_mean": target_dict_val["mfp_mean"],
     "afp": target_dict_val["afp"],
     "hybrid_dim0": target_dict_val["hybrid_dim0"],
     "hybrid_dim1": target_dict_val["hybrid_dim1"]}
)
record_val.index += 1
record_val.to_csv("record_transfer_activation_val.csv")

# for target in target_list:
#     index_min = np.argmin(target_dict[target])
#     print("="*45)
#     print("For {} model, the best model is:".format(target))
#     print("Fold {}:".format(index_min + 1))
#     print("Hyperparameter: {}".format(hyper_set[index_min]))
#     print("RMSE: {}".format(target_dict[target][index_min]))
#     print("="*45)