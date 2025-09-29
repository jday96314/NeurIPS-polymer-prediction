import polars as pl
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from cir_model import CenteredIsotonicRegression
import os

def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else None

def preprocess_dmitry():
    print('dmitry...')

    TARGET_NAME = 'Density'

    temp_df = (
        data_density
        .join(preds_df, on='SMILES', how='inner')
        .with_columns(
            pl.col(TARGET_NAME).alias(f'{TARGET_NAME}_label'),
            pl.col(f'{TARGET_NAME}_right').alias(f'{TARGET_NAME}_pred'),
        )
    )['SMILES', f'{TARGET_NAME}_label', f'{TARGET_NAME}_pred']

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{TARGET_NAME}_label'], temp_df[f'{TARGET_NAME}_pred'])
    rescaled_labels = model.predict(temp_df[f'{TARGET_NAME}_label'])
    temp_df = temp_df.with_columns(**{f'{TARGET_NAME}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_rescaled_label']))

    os.makedirs(f'data_preprocessing/results/{TARGET_NAME}', exist_ok=True)
    temp_df.write_csv(f'data_preprocessing/results/{TARGET_NAME}/dmitry.csv')


def preprocess_dmitry_2():
    print('dmitry 2...')

    TARGET_NAME = 'Tg'
    temp_df = (
        data_tg2
        .join(preds_df, on='SMILES', how='inner')
        .with_columns(
            pl.col(TARGET_NAME).alias(f'{TARGET_NAME}_label'),
            pl.col(f'{TARGET_NAME}_right').alias(f'{TARGET_NAME}_pred'),
        )
    )['SMILES', f'{TARGET_NAME}_label', f'{TARGET_NAME}_pred']

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{TARGET_NAME}_label'], temp_df[f'{TARGET_NAME}_pred'])
    rescaled_labels = model.predict(temp_df[f'{TARGET_NAME}_label'])
    temp_df = temp_df.with_columns(**{f'{TARGET_NAME}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_rescaled_label']))

    os.makedirs(f'data_preprocessing/results/{TARGET_NAME}', exist_ok=True)
    temp_df.write_csv(f'data_preprocessing/results/{TARGET_NAME}/dmitry_2.csv')

def preprocess_dmitry_3():
    print('dmitry 3...')

    TARGET_NAME = 'Tg'

    temp_df = (
        data_tg3
        .join(preds_df, on='SMILES', how='inner')
        .with_columns(
            pl.col(TARGET_NAME).alias(f'{TARGET_NAME}_label'),
            pl.col(f'{TARGET_NAME}_right').alias(f'{TARGET_NAME}_pred'),
        )
    )['SMILES', f'{TARGET_NAME}_label', f'{TARGET_NAME}_pred']

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{TARGET_NAME}_label'], temp_df[f'{TARGET_NAME}_pred'])
    rescaled_labels = model.predict(temp_df[f'{TARGET_NAME}_label'])
    temp_df = temp_df.with_columns(**{f'{TARGET_NAME}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_rescaled_label']))

    os.makedirs(f'data_preprocessing/results/{TARGET_NAME}', exist_ok=True)
    temp_df.write_csv(f'data_preprocessing/results/{TARGET_NAME}/dmitry_3.csv')

def preprocess_host_extra(target_name: str):
    print(f'host extra {target_name}...')

    temp_df = (
        host_labels_df
        .drop_nulls(subset=target_name)
        .join(host_preds_df, on='SMILES', how='inner')
        .with_columns(
            pl.col(target_name).alias(f'{target_name}_label').cast(float),
            pl.col(f'{target_name}_right').alias(f'{target_name}_pred'),
        )
    )['SMILES', f'{target_name}_label', f'{target_name}_pred']

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{target_name}_pred'], temp_df[f'{target_name}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{target_name}_label'], temp_df[f'{target_name}_pred'])
    rescaled_labels = model.predict(temp_df[f'{target_name}_label'])
    temp_df = temp_df.with_columns(**{f'{target_name}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{target_name}_pred'], temp_df[f'{target_name}_rescaled_label']))

    os.makedirs(f'data_preprocessing/results/{target_name}', exist_ok=True)
    temp_df.write_csv(f'data_preprocessing/results/{target_name}/host_extra.csv')

def preprocess_lamalab():
    print('lamalab...')

    TARGET_NAME = 'Tg'

    temp_df = (
        lamalab_tg_labels_df
        .join(preds_df, on='SMILES', how='inner')
        .with_columns(
            (pl.col('labels.Exp_Tg(K)') - 273.15).alias(f'{TARGET_NAME}_label'),
            pl.col(f'{TARGET_NAME}').alias(f'{TARGET_NAME}_pred'),
        )
    )['SMILES', f'{TARGET_NAME}_label', f'{TARGET_NAME}_pred']

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{TARGET_NAME}_label'], temp_df[f'{TARGET_NAME}_pred'])
    rescaled_labels = model.predict(temp_df[f'{TARGET_NAME}_label'])
    temp_df = temp_df.with_columns(**{f'{TARGET_NAME}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{TARGET_NAME}_pred'], temp_df[f'{TARGET_NAME}_rescaled_label']))

    os.makedirs(f'data_preprocessing/results/{TARGET_NAME}', exist_ok=True)
    temp_df.write_csv(f'data_preprocessing/results/{TARGET_NAME}/LAMALAB.csv')

def preprocess_radonpy(target_name: str, apply_filter: bool = False):
    print(f'radonpy {target_name}...')

    target_names_to_radonpy_cols = {
        'Density': 'density',
        'Tc': 'thermal_conductivity',
        'Rg': 'Rg',
    }
    radonpy_col_name = target_names_to_radonpy_cols[target_name]

    if not apply_filter:
        if target_name != 'Rg':
            temp_df = (
                radonpy_labels_df
                .join(preds_df, on='SMILES', how='inner')
                .with_columns(
                    pl.col(radonpy_col_name).alias(f'{target_name}_label'),
                    pl.col(f'{target_name}').alias(f'{target_name}_pred'),
                )
            )['SMILES', f'{target_name}_label', f'{target_name}_pred']
        else:
            temp_df = (
                radonpy_labels_df
                .join(preds_df, on='SMILES', how='inner')
                .with_columns(
                    pl.col(target_name).alias(f'{target_name}_label'),
                    pl.col(f'{target_name}_right').alias(f'{target_name}_pred'),
                )
            )['SMILES', f'{target_name}_label', f'{target_name}_pred']
    else:
        temp_df = (
            radonpy_labels_df
            .join(preds_df, on='SMILES', how='inner')
            .with_columns(
                pl.col(radonpy_col_name).alias(f'{target_name}_label'),
                pl.col(f'{target_name}').alias(f'{target_name}_pred'),
            )
            .filter(pl.col(f'{target_name}_label') < .402)
        )['SMILES', f'{target_name}_label', f'{target_name}_pred']

    temp_df = temp_df.drop_nulls()

    print('Baseline MAE:', mean_absolute_error(temp_df[f'{target_name}_pred'], temp_df[f'{target_name}_label']))

    model = CenteredIsotonicRegression()
    model.fit(temp_df[f'{target_name}_label'], temp_df[f'{target_name}_pred'])
    rescaled_labels = model.predict(temp_df[f'{target_name}_label'])
    temp_df = temp_df.with_columns(**{f'{target_name}_rescaled_label': rescaled_labels})

    print('Scaled MAE:', mean_absolute_error(temp_df[f'{target_name}_pred'], temp_df[f'{target_name}_rescaled_label']))
    
    os.makedirs(f'data_preprocessing/results/{target_name}', exist_ok=True)
    suffix = '' if not apply_filter else '_filtered'
    temp_df.write_csv(f'data_preprocessing/results/{target_name}/RadonPy{suffix}.csv')

if __name__ == "__main__":
    # LOAD DATA.
    preds_df = pl.read_csv('data_preprocessing/results/extra_smiles_relabeled.csv')

    host_labels_df = pl.read_csv('data/from_host/train_host_extra.csv')
    host_preds_df = pl.read_csv('data_preprocessing/results/train_host_extra.csv')

    lamalab_tg_labels_df = pd.read_csv('data/LAMALAB_CURATED_Tg_structured_polymerclass.csv')
    lamalab_tg_labels_df['SMILES'] = lamalab_tg_labels_df['PSMILES'].apply(lambda s: canonicalize_smiles(s))
    lamalab_tg_labels_df = pl.from_pandas(lamalab_tg_labels_df)

    radonpy_labels_df = pd.read_csv('data/PI1070.csv')
    radonpy_labels_df['SMILES'] = radonpy_labels_df['smiles'].apply(lambda s: canonicalize_smiles(s))
    radonpy_labels_df = pl.from_pandas(radonpy_labels_df)

    # https://springernature.figshare.com/articles/dataset/dataset_with_glass_transition_temperature/24219958?file=42507037
    data_tg2 = pd.read_csv('data/smiles_extra_data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
    data_tg2['SMILES'] = data_tg2['SMILES'].apply(lambda s: canonicalize_smiles(s))
    data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})

    # https://www.sciencedirect.com/science/article/pii/S2590159123000377#ec0005
    data_tg3 = pd.read_excel('data/smiles_extra_data/data_tg3.xlsx')
    data_tg3['SMILES'] = data_tg3['SMILES'].apply(lambda s: canonicalize_smiles(s))
    data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
    data_tg3['Tg'] = data_tg3['Tg'] - 273.15

    # https://github.com/Duke-MatSci/ChemProps
    data_density = pd.read_excel('data/smiles_extra_data/data_dnst1.xlsx')
    data_density = data_density.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
    data_density['SMILES'] = data_density['SMILES'].apply(lambda s: canonicalize_smiles(s))
    data_density = data_density[(data_density['SMILES'].notnull())&(data_density['Density'].notnull())&(data_density['Density'] != 'nylon')]
    data_density['Density'] = data_density['Density'].astype('float64')
    data_density['Density'] -= 0.118

    data_tg2 = pl.from_pandas(data_tg2)
    data_tg3 = pl.from_pandas(data_tg3)
    data_density = pl.from_pandas(data_density)

    # PREPROCESS DATA.
    preprocess_dmitry()
    preprocess_dmitry_2()
    preprocess_dmitry_3()

    for target_name in ['Tg', 'FFV', 'Tc']:
        preprocess_host_extra(target_name)

    preprocess_lamalab()

    for target_name in ['Tc', 'Density', 'Rg']:
        preprocess_radonpy(target_name, apply_filter=False)

    preprocess_radonpy('Tc', apply_filter=True)