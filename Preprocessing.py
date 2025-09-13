import pandas as pd
import argparse
import logging
import os
from tqdm import tqdm

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
    """One-hot encode an amino acid sequence.

    Args:
        sequence (str): Amino acid sequence.
        amino_acid_list (list): Ordered list of valid amino acids.

    Returns:
        list: List of one-hot encoded vectors or empty list if invalid.
    """
    if pd.isna(sequence) or not sequence or not all(c in AMINO_ACID_PROPERTIES['valid_amino_acids'] for c in sequence):
        logging.warning(f"Invalid sequence: {sequence}")
        return []
    return [[1 if aa == amino_acid_list[i] else 0 for i in range(len(amino_acid_list))] for aa in sequence]

def get_physicochemical_properties(sequence, properties):
    """Extract physicochemical properties for a sequence.

    Args:
        sequence (str): Amino acid sequence.
        properties (dict): Dictionary of property scales (e.g., hydrophobicity, charge).

    Returns:
        tuple: Lists of hydrophobicity, charge, and polarity values or empty lists if invalid.
    """
    if pd.isna(sequence) or not sequence or not all(c in AMINO_ACID_PROPERTIES['valid_amino_acids'] for c in sequence):
        logging.warning(f"Invalid sequence for properties: {sequence}")
        return [], [], []
    hydrophobicity = [properties['hydrophobicity'].get(aa, 0) for aa in sequence]
    charge = [properties['charge'].get(aa, 0) for aa in sequence]
    polarity = [properties['polarity'].get(aa, 0) for aa in sequence]
    return hydrophobicity, charge, polarity

def add_features_to_dataframe(df, amino_acid_list, properties):
    """Add one-hot encoding and physicochemical properties to valid rows in DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'sequence' and 'valid' columns.
        amino_acid_list (list): Ordered list of amino acids for one-hot encoding.
        properties (dict): Dictionary of physicochemical property scales.

    Returns:
        pd.DataFrame: Updated DataFrame with new columns.
    """
    df['one_hot_sequence'] = [[]] * len(df)
    df['hydrophobicity'] = [[]] * len(df)
    df['charge'] = [[]] * len(df)
    df['polarity'] = [[]] * len(df)

    # Process only valid rows with non-empty sequences
    valid_mask = (df['valid'] == True) & (df['sequence'].notna()) & (df['sequence'] != '')
    valid_df = df[valid_mask].copy()

    if not valid_df.empty:
        tqdm.pandas(desc="Adding features to valid proteins")
        valid_df['one_hot_sequence'] = valid_df['sequence'].progress_apply(
            lambda seq: one_hot_encode_sequence(seq, amino_acid_list)
        )
        valid_df[['hydrophobicity', 'charge', 'polarity']] = valid_df['sequence'].progress_apply(
            lambda seq: pd.Series(get_physicochemical_properties(seq, properties))
        )
        df.loc[valid_mask, ['one_hot_sequence', 'hydrophobicity', 'charge', 'polarity']] = valid_df[
            ['one_hot_sequence', 'hydrophobicity', 'charge', 'polarity']
        ]

    return df

def main():
    """Main function to process CSV and add features."""
    parser = argparse.ArgumentParser(description="Add one-hot encoding and physicochemical properties to CB513 dataset CSV")
    parser.add_argument("--input_csv", default="cb513_ready_dataset_ml.csv", help="Input CSV file path")
    parser.add_argument("--output_csv", default="cb513_ready_dataset_ml_with_features.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_csv):
        logging.error(f"Input CSV file not found: {args.input_csv}")
        print(f"❌ Error: Input CSV file '{args.input_csv}' not found.")
        return

    try:
        # Load CSV
        logging.info(f"Loading CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)

        # Validate required columns
        required_columns = ['protein_id', 'sequence', 'valid']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logging.error(f"Missing required columns: {missing}")
            print(f"❌ Error: Missing required columns: {missing}")
            return

        # Add features
        logging.info("Adding one-hot encoding and physicochemical properties")
        df = add_features_to_dataframe(df, AMINO_ACID_PROPERTIES['amino_acid_list'], AMINO_ACID_PROPERTIES)

        # Save output
        logging.info(f"Saving output to: {args.output_csv}")
        df.to_csv(args.output_csv, index=False)

        # Summary
        valid_count = sum(df['valid'] & df['one_hot_sequence'].apply(lambda x: len(x) > 0))
        print(f"🎉 Processing complete! Saved {len(df)} proteins to '{args.output_csv}' "
              f"({valid_count} with features added)")
        logging.info(f"Processed {len(df)} proteins, {valid_count} with features")

    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        print(f"❌ Error: {e}. Check add_features.log for details.")

if __name__ == "__main__":
    main()