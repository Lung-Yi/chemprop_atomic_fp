# Molecular Property Prediction
This repositiory is used to implement the aotmic fingerprint method in Chemprop.

Please refer to the paper:

Deep Learning-Based Increment Theory for Formation Enthalpy Predictions
https://pubs.acs.org/doi/full/10.1021/acs.jpca.2c04848


## Benchmark with the heat of formation prediction with the molecular sum-pooling and mean-pooling methods

### load the pretrained model, finetune on the high-quality dataset
#### atomic fingerprint method 
python train.py \
    --data_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_train_12.csv \
    --separate_val_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_val_12.csv \
    --separate_test_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_test_12.csv \
    --dataset_type regression \
    --checkpoint_dir save_models/feature_4/afp_6 \
    --save_dir save_models/feature_4/benchmark_models/afp_6 \
    --warmup_epochs 0 --max_lr 5e-4 --init_lr 1e-4 \
    --epochs 20 --final_lr 1e-5 --no_features_scaling \
    --dropout 0 --hidden_size 300 --ffn_num_layers 4 \
    --save_preds --fp_method atomic --activation PReLU --gpu 0
    
### molecular sum-pooling method
python train.py \
    --data_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_train_12.csv \
    --separate_val_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_val_12.csv \
    --separate_test_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_test_12.csv \
    --dataset_type regression \
    --checkpoint_dir save_models/feature_4/mfp_sum_6 \
    --save_dir save_models/feature_4/benchmark_models/mfp_sum_6 \
    --warmup_epochs 0 --max_lr 5e-4 --init_lr 1e-4 \
    --epochs 20 --final_lr 1e-5 --no_features_scaling \
    --dropout 0 --hidden_size 300 --ffn_num_layers 4 \
    --save_preds --fp_method molecular --aggregation sum --activation PReLU --gpu 0

### molecular mean-pooling method
python train.py \
    --data_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_train_12.csv \
    --separate_val_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_val_12.csv \
    --separate_test_path heat_formation_data/new_CCSD_exp/size_split/CCSD_exp_size_test_12.csv \
    --dataset_type regression \
    --checkpoint_dir save_models/feature_4/mfp_mean_8 \
    --save_dir save_models/feature_4/benchmark_models/mfp_mean_8 \
    --warmup_epochs 0 --max_lr 5e-4 --init_lr 1e-4 \
    --epochs 20 --final_lr 1e-5 --no_features_scaling \
    --dropout 0 --hidden_size 400 --ffn_num_layers 3 \
    --save_preds --fp_method molecular --activation --gpu 0 --aggregation mean

## The original Chemprop reposittory is:
https://github.com/chemprop/chemprop

### Citation:
@article{chen2022deep,
  title={Deep Learning-Based Increment Theory for Formation Enthalpy Predictions},
  author={Chen, Lung-Yi and Hsu, Ting-Wei and Hsiung, Tsai-Chen and Li, Yi-Pei},
  journal={The Journal of Physical Chemistry A},
  volume={126},
  number={41},
  pages={7548--7556},
  year={2022},
  publisher={ACS Publications}
}