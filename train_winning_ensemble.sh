# INSTALL DEPENDENCIES.
pip install -r requirements.txt

# Need torch >= 2.8 for blackwell GPUs, but autogluon "requires" torch 2.7,
# so we install a conflicting version that still works.
pip install --upgrade torch torchvision torchaudio

# DOWNLOAD & UNPACK DATA.
mkdir -p data
cd data

kaggle competitions download -c neurips-open-polymer-prediction-2025
kaggle datasets download -d jsday96/pi1m-pseudolabels
kaggle datasets download -d dmitryuarov/smiles-extra-data
kaggle datasets download -d jsday96/polymer-merged-extra-host-data
kaggle datasets download -d jsday96/md-simulation-results
wget https://zenodo.org/records/15210035/files/LAMALAB_CURATED_Tg_structured_polymerclass.csv
wget -O PI1070.csv https://raw.githubusercontent.com/RadonPy/RadonPy/develop/data/PI1070.csv

unzip neurips-open-polymer-prediction-2025.zip -d from_host
unzip pi1m-pseudolabels.zip -d PI1M_pseudolabels
unzip smiles-extra-data.zip -d smiles_extra_data
unzip polymer-merged-extra-host-data.zip -d from_host
unzip md-simulation-results.zip -d md_simulation_results

# PREPROCESS DATA.
mkdir -p models
cd models

kaggle datasets download -d jsday96/polymer-relabeling-models
unzip polymer-relabeling-models.zip

cd ..
python data_preprocessing/relabeling.py
python data_preprocessing/merge_updated_labels.py

python uni_mol/unimol_datasets/get_extra_data.py

# TRAIN.
python bert/pretrain.py
python bert/supervised_finetune.py
python simulations/train_metric_predictor.py
python tabular/train.py
python uni_mol/train.py