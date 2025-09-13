import pandas as pd
import argparse
import logging
import os
from tqdm import tqdm
import numpy as np
import ast

# Set up logging
logging.basicConfig(
    filename='add_features.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Physicochemical properties configuration
AMINO_ACID_PROPERTIES = {
    'valid_amino_acids': set('ACDEFGHIKLMNPQRSTVWY'),
    'amino_acid_list': list('ACDEFGHIKLMNPQRSTVWY'),
    'hydrophobicity': {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
        'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
        'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    },
    'charge': {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 1,
        'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0,
        'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    },
    'polarity': {
        'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 1,
        'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 1, 'P': 0, 'Q': 1,
        'R': 1, 'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
    }
}

def one_hot_encode_sequence(sequence, amino_acid_list):
    """One-hot encode an amino acid sequence."""
    if pd.isna(sequence) or not sequence or not all(c in AMINO_ACID_PROPERTIES['valid_amino_acids'] for c in sequence):
        logging.warning(f"Invalid sequence: {sequence}")
        return []
    return [[1 if aa == amino_acid_list[i] else 0 for i in range(len(amino_acid_list))] for aa in sequence]

def normalize_coordinates(coordinates):
    """Normalize 3D coordinates: center (mean=0) and scale (std=1)."""
    try:
        coords = np.array(coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
            logging.warning(f"Invalid coordinates format: {coordinates}")
            return []
        # Center: Subtract mean
        mean = np.mean(coords, axis=0)
        centered = coords - mean
        # Scale: Divide by standard deviation (avoid division by zero)
        std = np.std(centered, axis=0)
        std[std == 0] = 1  # Avoid division issues
        normalized = centered / std
        return normalized.tolist()
    except Exception as e:
        logging.warning(f"Error normalizing coordinates: {e}")
        return []

def get_physicochemical_properties(sequence, coordinates, properties):
    """Extract physicochemical properties and normalized coordinates for a sequence."""
    if pd.isna(sequence) or not sequence or not all(c in AMINO_ACID_PROPERTIES['valid_amino_acids'] for c in sequence):
        logging.warning(f"Invalid sequence for properties: {sequence}")
        return [], [], [], []
    hydrophobicity = [properties['hydrophobicity'].get(aa, 0) for aa in sequence]
    charge = [properties['charge'].get(aa, 0) for aa in sequence]
    polarity = [properties['polarity'].get(aa, 0) for aa in sequence]
    # Handle coordinates
    if coordinates is None or pd.isna(coordinates):
        logging.info("No coordinates provided, setting normalized_coordinates to empty")
        return hydrophobicity, charge, polarity, []
    try:
        coords = ast.literal_eval(coordinates) if isinstance(coordinates, str) else coordinates
        normalized_coords = normalize_coordinates(coords)
        if len(normalized_coords) != len(sequence):
            logging.warning(f"Coordinates length mismatch: {len(coords)} vs sequence {len(sequence)}")
            return hydrophobicity, charge, polarity, []
        return hydrophobicity, charge, polarity, normalized_coords
    except Exception as e:
        logging.warning(f"Error processing coordinates: {e}")
        return hydrophobicity, charge, polarity, []

def add_features_to_dataframe(df, amino_acid_list, properties, coord_dict=None):
    """Add one-hot encoding, physicochemical properties, and normalized coordinates to valid rows."""
    df['one_hot_sequence'] = [[]] * len(df)
    df['hydrophobicity'] = [[]] * len(df)
    df['charge'] = [[]] * len(df)
    df['polarity'] = [[]] * len(df)
    df['normalized_coordinates'] = [[]] * len(df)

    valid_mask = (df['valid'] == True) & (df['sequence'].notna()) & (df['sequence'] != '')
    valid_df = df[valid_mask].copy()

    if not valid_df.empty:
        tqdm.pandas(desc="Adding features to valid proteins")
        valid_df['one_hot_sequence'] = valid_df['sequence'].progress_apply(
            lambda seq: one_hot_encode_sequence(seq, amino_acid_list)
        )
        valid_df[['hydrophobicity', 'charge', 'polarity', 'normalized_coordinates']] = valid_df.apply(
            lambda row: pd.Series(get_physicochemical_properties(
                row['sequence'], 
                coord_dict.get(row['protein_id']) if coord_dict else row.get('coordinates'), 
                properties
            )),
            axis=1
        )
        df.loc[valid_mask, ['one_hot_sequence', 'hydrophobicity', 'charge', 'polarity', 'normalized_coordinates']] = valid_df[
            ['one_hot_sequence', 'hydrophobicity', 'charge', 'polarity', 'normalized_coordinates']
        ]

    return df

def build_window_features(one_hot, hydro, charge, polarity, coords, window_size=7):
    """Build windowed feature vectors, including normalized coordinates if available."""
    seq_len = len(one_hot)
    half_win = window_size // 2
    has_coords = len(coords) > 0
    feature_dim = len(one_hot[0]) + 3 + (3 if has_coords else 0)  # 20 + 1 + 1 + 1 + (3 if coords)
    padded_features = np.zeros((seq_len + 2 * half_win, feature_dim))
    features = np.hstack([
        one_hot,
        np.array(hydro)[:, np.newaxis],
        np.array(charge)[:, np.newaxis],
        np.array(polarity)[:, np.newaxis],
        np.array(coords) if has_coords else np.empty((seq_len, 0))
    ])
    padded_features[half_win:seq_len + half_win] = features
    
    windowed = np.zeros((seq_len, feature_dim * window_size))
    for i in range(seq_len):
        window = padded_features[i:i + window_size].flatten()
        windowed[i] = window
    return windowed

def load_dataset(input_csv, window_size=7):
    """Load dataset and create rich feature vectors for training."""
    df = pd.read_csv(input_csv)
    df = df[df['valid'] == True]

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        one_hot = np.array(ast.literal_eval(row['one_hot_sequence']))
        hydro = ast.literal_eval(row['hydrophobicity'])
        charge = ast.literal_eval(row['charge'])
        polarity = ast.literal_eval(row['polarity'])
        coords = ast.literal_eval(row['normalized_coordinates']) if row['normalized_coordinates'] else []
        labels = ast.literal_eval(row['numeric_labels'])
        
        windowed_features = build_window_features(one_hot, hydro, charge, polarity, coords, window_size)
        X_list.append(windowed_features)
        y_list.append(np.array(labels))

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def main():
    """Main function to process CSV and add features."""
    parser = argparse.ArgumentParser(description="Add one-hot encoding, physicochemical properties, and normalized coordinates to CB513 dataset CSV")
    parser.add_argument("--input_csv", default="cb513_ready_dataset_ml.csv", help="Input CSV file path")
    parser.add_argument("--output_csv", default="cb513_ready_dataset_ml_with_features.csv", help="Output CSV file path")
    parser.add_argument("--coord_file", help="Optional file with coordinates (protein_id: [[x,y,z], ...])")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logging.error(f"Input CSV file not found: {args.input_csv}")
        print(f"❌ Error: Input CSV file '{args.input_csv}' not found.")
        return

    try:
        logging.info(f"Loading CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)

        required_columns = ['protein_id', 'sequence', 'valid', 'numeric_labels']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Missing required columns: {missing}")
            print(f"❌ Error: Missing required columns: {missing}")
            return

        # Load coordinates if provided
        coord_dict = {}
        if args.coord_file and os.path.exists(args.coord_file):
            try:
                coord_df = pd.read_csv(args.coord_file)
                if 'protein_id' in coord_df.columns and 'coordinates' in coord_df.columns:
                    coord_dict = dict(zip(coord_df['protein_id'], coord_df['coordinates']))
                    logging.info(f"Loaded coordinates for {len(coord_dict)} proteins from {args.coord_file}")
                else:
                    logging.warning(f"Coordinate file {args.coord_file} missing required columns")
            except Exception as e:
                logging.warning(f"Error loading coordinate file {args.coord_file}: {e}")

        logging.info("Adding one-hot encoding, physicochemical properties, and normalized coordinates")
        df = add_features_to_dataframe(df, AMINO_ACID_PROPERTIES['amino_acid_list'], AMINO_ACID_PROPERTIES, coord_dict)

        logging.info(f"Saving output to: {args.output_csv}")
        df.to_csv(args.output_csv, index=False)

        valid_count = sum(df['valid'] & df['one_hot_sequence'].apply(lambda x: len(x) > 0))
        coord_count = sum(df['normalized_coordinates'].apply(lambda x: len(x) > 0))
        print(f"🎉 Processing complete! Saved {len(df)} proteins to '{args.output_csv}' "
              f"({valid_count} with features, {coord_count} with coordinates)")
        logging.info(f"Processed {len(df)} proteins, {valid_count} with features, {coord_count} with coordinates")

    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        print(f"❌ Error: {e}. Check add_features.log for details.")

if __name__ == "__main__":
    main()