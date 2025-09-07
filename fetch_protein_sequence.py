import pandas as pd
import os
import logging
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(
    filename='cb513_conversion.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Valid amino acids and DSSP labels
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
VALID_DSSP_LABELS = set('HEC')

# Load CB513 dataset from HuggingFace
url = "https://huggingface.co/datasets/proteinea/secondary_structure_prediction/resolve/main/CB513.csv"

try:
    print("⬇️ Attempting to download CB513 dataset...")
    df = pd.read_csv(url)
    print("✅ Dataset loaded successfully!")
    print(f"📊 Shape: {df.shape}")
    print(f"\n📑 Columns: {list(df.columns)}\n")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    print(f"❌ Failed to load dataset. Check cb513_conversion.log for details.")
    exit(1)

# Validate required columns
required_columns = ['input', 'dssp3']
if not all(col in df.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    logging.error(f"Missing required columns: {missing_cols}")
    print(f"❌ Missing required columns: {missing_cols}. Check cb513_conversion.log.")
    exit(1)

# Create output folders safely
try:
    os.makedirs("CB513_FASTA", exist_ok=True)
    os.makedirs("CB513_DSSP", exist_ok=True)
except Exception as e:
    logging.error(f"Failed to create output directories: {e}")
    print(f"❌ Failed to create output directories. Check cb513_conversion.log.")
    exit(1)

print("🔄 Converting to FASTA + DSSP format...")

for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing proteins"):
    try:
        protein_id = f"protein_{i+1:03d}"   # Auto-generate ID
        sequence = row["input"]
        labels = row["dssp3"]  # Use 3-class DSSP labels (H/E/C)

        # Validate sequence and labels
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Sequence is empty or not a string")
        if not labels or not isinstance(labels, str):
            raise ValueError("DSSP labels are empty or not a string")
        
        # Validate amino acids
        if not all(c in VALID_AMINO_ACIDS for c in sequence):
            invalid_chars = set(sequence) - VALID_AMINO_ACIDS
            raise ValueError(f"Invalid amino acids found: {invalid_chars}")
        
        # Validate DSSP labels
        if not all(c in VALID_DSSP_LABELS for c in labels):
            invalid_chars = set(labels) - VALID_DSSP_LABELS
            raise ValueError(f"Invalid DSSP labels found: {invalid_chars}")
        
        # Save FASTA file
        fasta_path = f"CB513_FASTA/{protein_id}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">{protein_id}\n")
            f.write(sequence + "\n")

        # Save DSSP label file
        dssp_path = f"CB513_DSSP/{protein_id}.dssp"
        with open(dssp_path, "w") as f:
            f.write(f"> {protein_id} Secondary Structure\n")
            f.write(labels + "\n")

    except Exception as e:
        logging.error(f"Failed to process protein {protein_id}: {e}")

print("🎉 Conversion complete!")
print("💾 FASTA files in CB513_FASTA/")
print("💾 DSSP label files in CB513_DSSP/")
print("📜 Check cb513_conversion.log for any processing errors.")