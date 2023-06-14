from rmgpy.data.thermo import ThermoDatabase
from rmgpy.species import Species
import argparse
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem

def GAVs_Hf298_cal(smiles, loaded_database):
    mol = Species().from_smiles(smiles)
    mol.generate_resonance_structures()
    data = loaded_database.get_thermo_data_from_groups(mol)
    Hf_298K = (data.H298.value) / 4.184  # unit: kcal/mol
    
    return Hf_298K

def ring_convert(x):
    if x == 0:
        return 'acyclic'
    elif x ==1:
        return 'monocyclic'
    else:
        return 'polycyclic'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_data',type=str,
                        default='../heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_test_12.csv')
    parser.add_argument('--save_dir',type=str,
                        default='../heat_formation_data/BGIT')
    args = parser.parse_args()

    
    database = ThermoDatabase()
    database.load_groups('/home/lungyi/anaconda3/envs/rmg_env/share/rmgdatabase/thermo/groups')

    data_gold = pd.read_csv(args.gold_data)
    smiles_list = list(data_gold['smiles'])
    hf_list = []

    mols = [Chem.MolFromSmiles(smi) for smi in list(data_gold['smiles'])]
    ring_number = [mol.GetRingInfo().NumRings() for mol in mols]
    ring_class = [ring_convert(x) for x in ring_number]
    data_gold['ring class'] = ring_class
    data_gold = pd.concat([data_gold.loc[data_gold['ring class'] == 'acyclic'],
                           data_gold.loc[data_gold['ring class'] == 'monocyclic'],
                           data_gold.loc[data_gold['ring class'] == 'polycyclic']]
                          , axis=0)

    for smiles in tqdm(list(data_gold['smiles'])):
        GAV_Hf_298K = GAVs_Hf298_cal(smiles, database)
        hf_list.append(GAV_Hf_298K)
    
    data_gold['BGIT hf 298K'] = hf_list
    data_gold['difference'] = [x-y for x,y in zip(list(data_gold['hf']), data_gold['BGIT hf 298K'] )]
    data_gold.to_csv(os.path.join(args.save_dir, 'BGIT_predictions.csv'))

    # # draw the plot
    min_point = min(min(data_gold['hf']), min(data_gold['BGIT hf 298K']))
    max_point = max(max(data_gold['hf']), max(data_gold['BGIT hf 298K']))
    points = np.linspace(min_point,max_point,20)
    
    plt.figure(figsize=(5,5), dpi=200)
    plt.plot(points, points, color='black', linestyle='--')
    sns.scatterplot(data=data_gold, x='hf', y='BGIT hf 298K',
                    hue = 'ring class', palette=sns.color_palette("hls", 3),)
    plt.xlabel('True heat of formation (kcal/mol)', fontsize=12)
    plt.ylabel('Predicted heat of formation (kcal/mol)', fontsize=12)
    plt.savefig(os.path.join(args.save_dir, 'true_prediction_plot_BGIT.png'), bbox_inches='tight')
    
    # Calculate RMSE and MAE
    MAE = dict()
    RMSE = dict()
    MAE.update({'overall': np.mean([np.abs(d) for d in list(data_gold['difference'])])})
    RMSE.update({'overall': (np.mean([d**2 for d in list(data_gold['difference'])]))**(1/2)})
    print('Overall Performance:')
    print('RMSE: ',RMSE['overall'])
    print('MAE: ', MAE['overall'])
    f = open(os.path.join(args.save_dir, 'performance_BGIT.txt'), 'w')
    f.write('Overall:\n')
    f.write('RMSE,MAE\n{:.4f},{:.4f}\n'.format(RMSE['overall'], MAE['overall']))
    
    for ring in ['acyclic', 'monocyclic', 'polycyclic']:
        diff = list(data_gold.loc[data_gold['ring class'] == ring]['difference'])
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

