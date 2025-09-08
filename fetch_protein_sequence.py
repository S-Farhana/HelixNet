import os
import pandas as pd
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename='cb513_dataset_parser.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Valid amino acids and DSSP labels
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
VALID_DSSP_LABELS = set('HEC')
LABEL_MAP = {'H':0, 'E':1, 'C':2}  # Numeric mapping for ML

# Directories
FASTA_DIR = "CB513_FASTA"
DSSP_DIR = "CB513_DSSP"
PDB_DIR = None  # Set to a directory path containing PDB files, or None to skip
OUTPUT_CSV = "cb513_ready_dataset_ml.csv"

# If CSV already exists, remove it to avoid duplicates
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

parsed_data = []

def parse_fasta(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None, None, "Error: FASTA file is empty"
            header = lines[0].strip()
            if not header.startswith('>'):
                return None, None, "Error: Invalid FASTA header"
            protein_id = header[1:].split()[0]
            sequence = ''.join(line.strip() for line in lines[1:])
            if not sequence:
                return protein_id, None, "Error: Sequence is empty"
            if not all(c in VALID_AMINO_ACIDS for c in sequence):
                invalid_chars = set(sequence) - VALID_AMINO_ACIDS
                return protein_id, sequence, f"Error: Invalid amino acids: {invalid_chars}"
            return protein_id, sequence, "Valid"
    except Exception as e:
        logging.error(f"Failed to parse FASTA {file_path}: {e}")
        return None, None, f"Error: {e}"

def parse_dssp(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None, None, "Error: DSSP file is empty"
            header = lines[0].strip()
            if not header.startswith('>'):
                return None, None, "Error: Invalid DSSP header"
            protein_id = header[1:].split()[0]
            labels = ''.join(line.strip() for line in lines[1:])
            if not labels:
                return protein_id, None, "Error: DSSP labels are empty"
            if not all(c in VALID_DSSP_LABELS for c in labels):
                invalid_chars = set(labels) - VALID_DSSP_LABELS
                return protein_id, labels, f"Error: Invalid DSSP labels: {invalid_chars}"
            return protein_id, labels, "Valid"
    except Exception as e:
        logging.error(f"Failed to parse DSSP {file_path}: {e}")
        return None, None, f"Error: {e}"

def parse_pdb(file_path, sequence_length):
    try:
        labels = ['C'] * sequence_length
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('HELIX'):
                    try:
                        start_res = int(line[21:25].strip())
                        end_res = int(line[33:37].strip())
                        for i in range(max(0, start_res-1), min(end_res, sequence_length)):
                            labels[i] = 'H'
                    except:
                        continue
                elif line.startswith('SHEET'):
                    try:
                        start_res = int(line[22:26].strip())
                        end_res = int(line[33:37].strip())
                        for i in range(max(0, start_res-1), min(end_res, sequence_length)):
                            labels[i] = 'E'
                    except:
                        continue
        labels_str = ''.join(labels)
        if len(labels_str) != sequence_length:
            return labels_str, f"Error: Label length mismatch ({len(labels_str)} != {sequence_length})"
        return labels_str, "Valid"
    except Exception as e:
        logging.error(f"Failed to parse PDB {file_path}: {e}")
        return None, f"Error: {e}"

print("🔄 Parsing FASTA and DSSP/PDB files...")

fasta_files = [f for f in os.listdir(FASTA_DIR) if f.endswith('.fasta')]
if not fasta_files:
    logging.error("No FASTA files found in CB513_FASTA directory")
    print("❌ No FASTA files found. Check cb513_dataset_parser.log.")
    exit(1)

for fasta_file in tqdm(fasta_files, desc="Parsing proteins"):
    try:
        fasta_path = os.path.join(FASTA_DIR, fasta_file)
        protein_id, sequence, fasta_status = parse_fasta(fasta_path)
        if not protein_id:
            parsed_data.append({
                'protein_id': fasta_file.replace('.fasta',''),
                'sequence': '',
                'labels': '',
                'numeric_labels': [],
                'status': fasta_status
            })
            continue

        sequence_length = len(sequence)
        record = {
            'protein_id': protein_id,
            'sequence': sequence,
            'labels': '',
            'numeric_labels': [],
            'status': fasta_status
        }

        # DSSP or PDB parsing
        dssp_file = f"{protein_id}.dssp"
        dssp_path = os.path.join(DSSP_DIR, dssp_file)
        if os.path.exists(dssp_path):
            _, labels, dssp_status = parse_dssp(dssp_path)
            if labels and len(labels) == sequence_length:
                record['labels'] = labels
                record['numeric_labels'] = [LABEL_MAP[c] for c in labels]
                record['status'] = "Valid"
            else:
                record['status'] = f"Error: DSSP length mismatch or invalid ({dssp_status})"
        elif PDB_DIR:
            pdb_file = os.path.join(PDB_DIR, f"{protein_id}.pdb")
            if os.path.exists(pdb_file):
                labels, pdb_status = parse_pdb(pdb_file, sequence_length)
                if labels and len(labels) == sequence_length:
                    record['labels'] = labels
                    record['numeric_labels'] = [LABEL_MAP[c] for c in labels]
                    record['status'] = "Valid"
                else:
                    record['status'] = f"Error: PDB parsing failed ({pdb_status})"
            else:
                record['status'] = "Error: DSSP/PDB not found"
        else:
            record['status'] = "Error: DSSP not found, PDB parsing disabled"

        parsed_data.append(record)

    except Exception as e:
        logging.error(f"Error processing {fasta_file}: {e}")
        parsed_data.append({
            'protein_id': fasta_file.replace('.fasta',''),
            'sequence': '',
            'labels': '',
            'numeric_labels': [],
            'status': f"Error: {e}"
        })

# Overwrite CSV every run to avoid multiple files
ml_data = [r for r in parsed_data if r['status'] == "Valid"]
df = pd.DataFrame(ml_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 ML-ready dataset saved! {len(ml_data)} proteins to {OUTPUT_CSV}")
print(f"📊 Total proteins processed: {len(parsed_data)}")
