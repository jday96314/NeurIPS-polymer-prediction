import glob
from xgboost import XGBRegressor
import polars as pl
import optuna
import json
import datetime
from sklearn.metrics import root_mean_squared_error
import os
from scipy.stats import pearsonr
from glob import glob
import joblib
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('./')
sys.path.append('./tabular')
from tabular.train import get_features_dataframe

def get_optimal_config(
        train_features_df: pl.DataFrame, 
        train_labels: pl.Series, 
        test_features_df: pl.DataFrame, 
        test_labels: pl.Series,
        trail_count: int,
        target_name: str,
        output_directory_path: str):
    def objective(trial: optuna.Trial) -> float:
        param = {
            'objective': 'reg:squarederror',
            'device': 'cuda',
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 0.3, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 20.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'max_depth': trial.suggest_int('max_depth', 3, 11),
        }

        model = XGBRegressor(**param)
        model.fit(
            train_features_df.to_numpy(), 
            train_labels.to_numpy(), 
            verbose=False
        )
        preds = model.predict(test_features_df.to_numpy())
        rmse = root_mean_squared_error(test_labels.to_numpy(), preds)
        print(pearsonr(test_labels.to_numpy(), preds))
        return rmse
    

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trail_count)

    with open(f'{output_directory_path}/{target_name}.json', 'w') as output_file:
        json.dump(study.best_params, output_file, indent=4)

    print(f"\nBest RMSE: {study.best_value:.5f}")
    print("Params:")
    print(json.dumps(study.best_params, indent=4))

if __name__ == '__main__':
    # LOAD DATA.
    raw_train_df = pl.read_csv('data/md_simulation_results/PI1M_hybrid_09_07.csv')
    # raw_train_df = pl.read_csv('simulations/generated_metrics/PI1M_hybrid_09_13.csv')
    raw_test_df = pl.read_csv('data/md_simulation_results/host_hybrid.csv')

    # COMPUTE FEATURES
    FEATURES_CONFIG = {
        'morgan_fingerprint_dim': 2048,
        'atom_pair_fingerprint_dim': 0,
        'torsion_dim': 256,
        'use_maccs_keys': True,
        'use_graph_features': True,
        'backbone_sidechain_detail_level': 2,
    }
    with open('simulations/models/features_config.json', 'w') as features_config_file:
        json.dump(FEATURES_CONFIG, features_config_file, indent=4)

    train_features_df: pl.DataFrame = get_features_dataframe(
        smiles_df=raw_train_df[['SMILES']], 
        **FEATURES_CONFIG
    )
    test_features_df: pl.DataFrame = get_features_dataframe(
        smiles_df=raw_test_df[['SMILES']],
        **FEATURES_CONFIG
    )

    '''
    # CREATE OUTPUT DIRECTORY.
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_directory_path = f'simulations/configs/metric_predictor_{timestamp}'
    os.makedirs(output_directory_path, exist_ok=True)

    # OPTIMIZE FOR EACH TARGET.
    target_names = [col_name for col_name in raw_train_df.columns if col_name not in ['SMILES', 'selection', 'grid_spacing_A']]
    for target_name in target_names:
        print(f'\nOptimizing for target: {target_name}')
        train_labels = raw_train_df[target_name]
        test_labels = raw_test_df[target_name]

        get_optimal_config(
            train_features_df=train_features_df, 
            train_labels=train_labels, 
            test_features_df=test_features_df, 
            test_labels=test_labels,
            trail_count=30,
            target_name=target_name,
            output_directory_path=output_directory_path
        )
    '''

    config_paths = glob('simulations/configs/metric_predictor_20250902_074142/*.json')
    pearson_rs = []
    for config_path in tqdm(config_paths):
        target_name = os.path.basename(config_path).replace('.json', '')
        # print(f'\nTraining final model for target: {target_name}')
        train_labels = raw_train_df[target_name]
        test_labels = raw_test_df[target_name]
        
        if train_labels.std() < 1e-4:
            continue

        with open(config_path, 'r') as config_file:
            best_params = json.load(config_file)
        
        model = XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            device='cuda',
            tree_method='hist',
        )
        model.fit(
            train_features_df.to_numpy(), 
            train_labels.to_numpy(), 
            verbose=True
        )
        
        preds = model.predict(test_features_df.to_numpy())
        rmse = root_mean_squared_error(test_labels.to_numpy(), preds)
        r, p = pearsonr(test_labels.to_numpy(), preds)
        pearson_rs.append(r)

        # print(f'Final RMSE: {rmse:.5f}')
        # print(f'Pearson r: {r:.5f}, p: {p:.5e}')

        if r < 0 or p > 0.05:
            # print('Skipping saving model due to poor performance.')
            continue

        os.makedirs('simulations/models', exist_ok=True)
        joblib.dump(model, f'simulations/models/xgb_{target_name}.joblib')

    print('\nAverage Pearson r:', np.mean(pearson_rs))