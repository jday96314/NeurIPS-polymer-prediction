import os
import glob
import pickle
from multiprocessing import get_context, cpu_count
from tqdm import tqdm
import warnings

import polars as pl
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

import pyarrow as pa
import pyarrow.parquet as pq

# Threshold for class-1 probability to consider a SMILES realistic
REALISTIC_PROBABILITY_THRESHOLD = 0.0056

# Paths
INPUT_GLOB_PATTERN = 'data/polyOne/polyOne_*.parquet'
OUTPUT_BASE_DIRECTORY = 'data/polyOne_partitioned_mini'
REALISTIC_SUBDIRECTORY = os.path.join(OUTPUT_BASE_DIRECTORY, 'realistic')
UNREALISTIC_SUBDIRECTORY = os.path.join(OUTPUT_BASE_DIRECTORY, 'unrealistic')
MODEL_FILE_PATH = 'property_tokenization/filtering_model.pkl'

# Batch size
BATCH_SIZE = 10_000

# Load pretrained classifier once
warnings.filterwarnings("ignore", category=UserWarning)
with open(MODEL_FILE_PATH, 'rb') as model_file:
    classifier_model = pickle.load(model_file)

# Fingerprint generators once
morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
atom_pair_fingerprint_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

def process_file(file_path: str) -> None:
    """
    Load SMILES from a parquet file, predict probabilities in batches,
    partition into realistic vs unrealistic, and stream results to two Parquet files.
    """
    # Read input parquet
    smiles_dataframe = pl.read_parquet(file_path).sample(fraction=0.05)
    total_rows = smiles_dataframe.height

    # Prepare output paths & ensure dirs
    base_filename = os.path.basename(file_path)
    realistic_output_path = os.path.join(REALISTIC_SUBDIRECTORY, base_filename)
    unrealistic_output_path = os.path.join(UNREALISTIC_SUBDIRECTORY, base_filename)
    os.makedirs(REALISTIC_SUBDIRECTORY, exist_ok=True)
    os.makedirs(UNREALISTIC_SUBDIRECTORY, exist_ok=True)

    # Initialize ParquetWriters to None; we'll create them on first batch
    realistic_writer = None
    unrealistic_writer = None

    # Process in batches
    for batch_start in tqdm(range(0, total_rows, BATCH_SIZE)):
        batch_df = smiles_dataframe.slice(batch_start, BATCH_SIZE)
        smiles_list = batch_df.get_column('smiles').to_list()

        # Compute fingerprint matrix for this batch
        fingerprint_list = []
        for smiles_string in smiles_list:
            mol = Chem.MolFromSmiles(smiles_string)
            morgan_fp = morgan_fingerprint_generator.GetFingerprint(mol)
            atom_pair_fp = atom_pair_fingerprint_generator.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            combined_array = np.concatenate([
                np.array(morgan_fp, dtype=np.uint8),
                np.array(atom_pair_fp, dtype=np.uint8),
                np.array(maccs_fp, dtype=np.uint8),
            ])
            fingerprint_list.append(combined_array)
        fingerprint_matrix = np.stack(fingerprint_list, axis=0)

        # Predict class-1 probabilities
        probabilities = classifier_model.predict_proba(fingerprint_matrix)[:, 1]

        # Attach to Polars batch and partition
        batch_with_probs = batch_df.with_columns(
            pl.Series('predicted_probability', probabilities)
        )
        realistic_batch = batch_with_probs.filter(
            pl.col('predicted_probability') > REALISTIC_PROBABILITY_THRESHOLD
        )
        unrealistic_batch = batch_with_probs.filter(
            pl.col('predicted_probability') <= REALISTIC_PROBABILITY_THRESHOLD
        )

        # Convert to Arrow tables
        realistic_table = realistic_batch.to_arrow()
        unrealistic_table = unrealistic_batch.to_arrow()

        # On first batch, create writers with schema
        if realistic_writer is None:
            realistic_writer = pq.ParquetWriter(
                realistic_output_path,
                realistic_table.schema,
                use_dictionary=True,
                compression='snappy'
            )
        if unrealistic_writer is None:
            unrealistic_writer = pq.ParquetWriter(
                unrealistic_output_path,
                unrealistic_table.schema,
                use_dictionary=True,
                compression='snappy'
            )

        # Write the batch
        if len(realistic_batch):
            realistic_writer.write_table(realistic_table)
        if len(unrealistic_batch):
            unrealistic_writer.write_table(unrealistic_table)

    # Close writers
    if realistic_writer:
        realistic_writer.close()
    if unrealistic_writer:
        unrealistic_writer.close()


def main() -> None:
    input_file_paths = glob.glob(INPUT_GLOB_PATTERN)
    with get_context('spawn').Pool(processes=cpu_count()) as pool:
        pool.map(process_file, input_file_paths)


if __name__ == '__main__':
    main()
