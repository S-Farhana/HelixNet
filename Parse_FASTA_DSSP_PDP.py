import os
import pandas as pd
import logging
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    filename='cb513_dataset_parser.log',
    level=logging.INFO,  # Capture INFO + ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Valid amino acids and DSSP labels
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
VALID_DSSP_LABELS = set('HEC')
LABEL_MAP = {'H': 0, 'E': 1, 'C': 2}  # Numeric mapping for ML

# Three-letter to one-letter amino acid mapping for PDB parsing
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def parse_fasta(file_path):
    """Parse FASTA file and return protein_id, sequence, status."""
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
    """Parse DSSP file and return protein_id, labels, status."""
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

def parse_pdb(file_path, fasta_sequence):
    """Parse PDB file with multi-chain and insertion code support."""
    try:
        chain_sequences = {}
        chain_residue_maps = {}

        # Extract CA atom sequences per chain
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    if atom_name != 'CA':
                        continue
                    chain_id = line[21].strip()
                    res_name = line[17:20].strip()
                    res_num = line[22:27].strip()  # includes insertion codes
                    if res_name not in AA_MAP:
                        continue
                    aa = AA_MAP[res_name]
                    if chain_id not in chain_sequences:
                        chain_sequences[chain_id] = []
                        chain_residue_maps[chain_id] = []
                    chain_sequences[chain_id].append(aa)
                    chain_residue_maps[chain_id].append(res_num)

        # Convert lists to strings
        for chain_id in chain_sequences:
            chain_sequences[chain_id] = ''.join(chain_sequences[chain_id])

        # Select best matching chain
        selected_chain = None
        if args.target_chain and args.target_chain in chain_sequences:
            selected_chain = args.target_chain
        else:
            max_similarity = 0
            for chain_id, pdb_seq in chain_sequences.items():
                min_len = min(len(pdb_seq), len(fasta_sequence))
                matches = sum(1 for a, b in zip(pdb_seq[:min_len], fasta_sequence[:min_len]) if a == b)
                similarity = matches / min_len if min_len > 0 else 0
                if similarity > max_similarity:
                    max_similarity = similarity
                    selected_chain = chain_id
            if not selected_chain:
                return None, "Error: No matching chain found"

        pdb_sequence = chain_sequences[selected_chain]
        residue_numbers = chain_residue_maps[selected_chain]
        sequence_length = len(fasta_sequence)
        labels = ['C'] * sequence_length  # default coil

        # Parse HELIX/SHEET for selected chain
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('HELIX'):
                    chain_id = line[19].strip()
                    if chain_id != selected_chain:
                        continue
                    start_res = line[21:26].strip()
                    end_res = line[33:38].strip()
                    if start_res in residue_numbers and end_res in residue_numbers:
                        start_idx = residue_numbers.index(start_res)
                        end_idx = residue_numbers.index(end_res) + 1
                        for i in range(start_idx, end_idx):
                            if i < sequence_length:
                                labels[i] = 'H'
                elif line.startswith('SHEET'):
                    chain_id = line[21].strip()
                    if chain_id != selected_chain:
                        continue
                    start_res = line[22:27].strip()
                    end_res = line[33:38].strip()
                    if start_res in residue_numbers and end_res in residue_numbers:
                        start_idx = residue_numbers.index(start_res)
                        end_idx = residue_numbers.index(end_res) + 1
                        for i in range(start_idx, end_idx):
                            if i < sequence_length:
                                labels[i] = 'E'

        labels_str = ''.join(labels)
        if len(labels_str) != sequence_length:
            return labels_str, f"Error: Label length mismatch ({len(labels_str)} != {sequence_length})"
        if not all(c in VALID_DSSP_LABELS for c in labels_str):
            invalid_chars = set(labels_str) - VALID_DSSP_LABELS
            return labels_str, f"Error: Invalid labels: {invalid_chars}"
        return labels_str, "Valid"
    except Exception as e:
        logging.error(f"Failed to parse PDB {file_path}: {e}")
        return None, f"Error: {e}"

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CB513 dataset into ML-ready CSV")
    parser.add_argument("--fasta_dir", required=True, help="Directory with FASTA files")
    parser.add_argument("--dssp_dir", required=True, help="Directory with DSSP files")
    parser.add_argument("--pdb_dir", default=None, help="Optional directory with PDB files")
    parser.add_argument("--output_csv", default="cb513_ready_dataset_ml.csv", help="Output CSV filename")
    parser.add_argument("--target_chain", default=None, help="Optional PDB chain ID (e.g., 'A')")
    args = parser.parse_args()

    # Remove existing CSV
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)
        logging.info(f"Removed existing {args.output_csv}")

    parsed_data = []

    # Main parsing loop
    print("🔄 Parsing FASTA and DSSP/PDB files for all proteins...")

    fasta_files = [f for f in os.listdir(args.fasta_dir) if f.endswith('.fasta')]
    if not fasta_files:
        logging.error("No FASTA files found in input directory")
        print("❌ No FASTA files found. Check cb513_dataset_parser.log.")
        exit(1)

    for fasta_file in tqdm(fasta_files, desc="Parsing proteins"):
        try:
            fasta_path = os.path.join(args.fasta_dir, fasta_file)
            protein_id, sequence, fasta_status = parse_fasta(fasta_path)
            if not protein_id:
                parsed_data.append({
                    'protein_id': fasta_file.replace('.fasta', ''),
                    'sequence': '',
                    'labels': '',
                    'numeric_labels': [],
                    'valid': False,
                    'status': fasta_status
                })
                continue

            sequence = sequence if sequence else ''
            sequence_length = len(sequence)
            record = {
                'protein_id': protein_id,
                'sequence': sequence,
                'labels': '',
                'numeric_labels': [],
                'valid': False,
                'status': fasta_status
            }

            # Try DSSP first
            dssp_file = f"{protein_id}.dssp"
            dssp_path = os.path.join(args.dssp_dir, dssp_file)
            if os.path.exists(dssp_path):
                dssp_protein_id, labels, dssp_status = parse_dssp(dssp_path)
                if dssp_protein_id != protein_id:
                    record['status'] = f"Error: Mismatched protein ID in DSSP (expected {protein_id}, got {dssp_protein_id})"
                elif labels and sequence and len(labels) != sequence_length:
                    record['status'] = f"Error: DSSP length mismatch (sequence {sequence_length}, labels {len(labels)})"
                else:
                    record['labels'] = labels if labels else ''
                    record['numeric_labels'] = [LABEL_MAP[c] for c in labels] if labels else []
                    record['valid'] = (dssp_status == "Valid" and fasta_status == "Valid")
                    record['status'] = dssp_status if dssp_status == "Valid" else f"Error: {dssp_status}"
            elif args.pdb_dir and sequence:
                pdb_file = os.path.join(args.pdb_dir, f"{protein_id}.pdb")
                if os.path.exists(pdb_file):
                    labels, pdb_status = parse_pdb(pdb_file, sequence)
                    if labels and len(labels) == sequence_length:
                        record['labels'] = labels
                        record['numeric_labels'] = [LABEL_MAP[c] for c in labels]
                        record['valid'] = (pdb_status == "Valid" and fasta_status == "Valid")
                        record['status'] = pdb_status if pdb_status == "Valid" else f"Error: {pdb_status}"
                    else:
                        record['status'] = f"Error: PDB parsing failed ({pdb_status})"
                else:
                    record['status'] = "Error: DSSP and PDB files not found"
            else:
                record['status'] = "Error: DSSP not found, PDB parsing disabled"

            parsed_data.append(record)

        except Exception as e:
            logging.error(f"Error processing {fasta_file}: {e}")
            parsed_data.append({
                'protein_id': fasta_file.replace('.fasta', ''),
                'sequence': '',
                'labels': '',
                'numeric_labels': [],
                'valid': False,
                'status': f"Error: {e}"
            })

    # Save all proteins to CSV
    if parsed_data:
        df = pd.DataFrame(parsed_data)
        df.to_csv(args.output_csv, index=False)
        valid_count = sum(df['valid'])
        print(f"🎉 Parsing complete! Saved {len(parsed_data)} proteins to {args.output_csv} ({valid_count} valid for ML)")
    else:
        print("❌ No proteins parsed. Check cb513_dataset_parser.log for details.")

    print(f"📊 Total proteins processed: {len(parsed_data)}")
    print("📜 Check cb513_dataset_parser.log for any parsing errors.")