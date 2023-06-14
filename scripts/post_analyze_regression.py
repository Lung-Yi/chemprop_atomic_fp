import pandas as pd
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

def ring_convert(x):
    if x == 0:
        return 'acyclic'
    elif x == 1:
        return 'monocyclic'
    else:
        return 'polycyclic'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_data',type=str,
                        default='../heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_test_12.csv')
    parser.add_argument('--model_dir',type=str,required=True)
    args = parser.parse_args()
    
    predict_path = os.path.join(args.model_dir, 'test_preds.csv')
    data_1 = pd.read_csv(args.gold_data)
    data_2 = pd.read_csv(predict_path)
    data_2 = data_2.rename(columns={'hf': 'hf_predict'})
    data = pd.concat([data_1, data_2[['hf_predict']]], axis=1)
    
    difference = [ data['hf'].loc[i] - data['hf_predict'].loc[i] for i in range(len(data)) ]
    mols = [Chem.MolFromSmiles(smi) for smi in list(data['smiles'])]
    ring_number = [mol.GetRingInfo().NumRings() for mol in mols]
    ring_class = [ring_convert(x) for x in ring_number]
    data['difference'] = difference
    data['ring class'] = ring_class

    # Calculate RMSE and MAE
    MAE = dict()
    RMSE = dict()
    MAE.update({'overall': np.mean([np.abs(d) for d in difference])})
    RMSE.update({'overall': (np.mean([d**2 for d in difference]))**(1/2)})
    
    data = pd.concat([data.loc[data['ring class'] == 'acyclic'],
                  data.loc[data['ring class'] == 'monocyclic'],
                  data.loc[data['ring class'] == 'polycyclic']],
                  axis=0)
    
    # draw the plot
    min_point = min(min(data['hf']), min(data['hf_predict']))
    max_point = max(max(data['hf']), max(data['hf_predict']))
    points = np.linspace(min_point,max_point,20)
    
    plt.figure(figsize=(4.5,4.5), dpi=800)
    plt.plot(points, points, color='black', linestyle='--')
    sns.scatterplot(data=data, x='hf', y='hf_predict',
                    hue = 'ring class', palette=sns.color_palette("hls", 3),)
    plt.xlabel('Reference heat of formation (kcal/mol)', fontsize=14)
    plt.ylabel('Predicted heat of formation (kcal/mol)', fontsize=14)
    plt.text(30, -280, "RMSE:{:.2f} kcal/mol\nMAE:  {:.2f} kcal/mol".format(RMSE['overall'], MAE['overall']),fontsize=11)
    plt.savefig(os.path.join(args.model_dir, 'true_prediction_plot.png'), bbox_inches='tight')
    

    print('Overall Performance:')
    print('RMSE: ',RMSE['overall'])
    print('MAE: ', MAE['overall'])
    f = open(os.path.join(args.model_dir, 'performance.txt'), 'w')
    f.write('Overall:\n')
    f.write('RMSE,MAE\n{:.4f},{:.4f}\n'.format(RMSE['overall'], MAE['overall']))
    
    for ring in ['acyclic', 'monocyclic', 'polycyclic']:
        diff = list(data.loc[data['ring class'] == ring]['difference'])
        mae = np.mean([np.abs(d) for d in diff])
        rmse = (np.mean([d**2 for d in diff]))**(1/2)
        MAE.update({ring: mae})
        RMSE.update({ring: rmse})
        print('{} Performance:'.format(ring))
        print('RMSE: ',RMSE[ring])
        print('MAE: ', MAE[ring])
        f.write('{}:\n'.format(ring))
        f.write('RMSE,MAE\n{:.4f},{:.4f}\n'.format(RMSE[ring], MAE[ring]))

    f.close()