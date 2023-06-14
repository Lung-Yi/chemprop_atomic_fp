import pandas as pd
import numpy as np
from os.path import join
import os, fnmatch
import argparse

def findSome(nPath, fTypes):
    allFiles = []
    for root, dirs, files in os.walk(nPath):
        for file in files:
            if file.endswith(fTypes):
                # print(os.path.join(root, file))
                if 'model_0' in dirs:
                    allFiles.append(os.path.join(root, file))
    return allFiles

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',type=str, required=True)
    args = parser.parse_args()

    files = findSome(args.model_dir, '.csv')[:] # except for the root test_preds.csv and test_scores.csv
    # print(files)
    new_data = pd.read_csv(files[0])
    new_data.rename(columns = {'hf':'hf_1'}, inplace = True)
    for j, file in enumerate(files[1:]):
        subdf = pd.read_csv(file)
        new_data['hf_{}'.format(j+2)] = subdf['hf']
    
    new_data['hf'] = np.mean(new_data[['hf_{}'.format(x) for x in range(1,11)]].values, axis=1)
    new_data.to_csv(os.path.join(args.model_dir, 'test_preds.csv'), index=False)




