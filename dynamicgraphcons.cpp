import os
import pandas as pd
import numpy as np
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data

# Ensure processed directory exists
os.makedirs("data/processed", exist_ok=True)

# Load metadata manifest if not already in memory
if 'df_meta' not in locals():
    df_meta = pd.read_csv("data/raw/metadata.csv")

parser = PDBParser(QUIET=True)

print("Starting feature extraction and dynamic graph construction for 30 proteins...")
success_count = 0

for idx, row in df_meta.iterrows():
    pdb_id = row['protein_id']
    struct_path = row['structure_path']

    if not os.path.exists(struct_path):
        print(f"Skipping {pdb_id}: File not found at {struct_path}")
        continue

    try:
        structure = parser.get_structure(pdb_id, struct_path)

        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
            break  # Take the first model only

        coords = np.array(coords)
        if len(coords) < 10:
            print(f"Skipping {pdb_id}: Too few residues ({len(coords)})")
            continue

        # Compute pseudo-dynamics features from static coordinates
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        rmsf = np.std(dist_matrix, axis=1)  # Pseudo-flexibility proxy
        dccm = np.corrcoef(dist_matrix)     # Pseudo-coupling proxy
        dccm = np.nan_to_num(dccm)

        # Build graph components
        src, dst = np.where((dist_matrix <= 8.0) & (dist_matrix > 0))
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        edge_attr = torch.tensor(np.stack([dist_matrix[src, dst], dccm[src, dst]], axis=-1), dtype=torch.float)
        x = torch.tensor(np.concatenate([rmsf[:, None], coords], axis=-1), dtype=torch.float)

        # Package into PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(data, f"data/processed/{pdb_id}_graph.pt")
        success_count += 1
        print(f"Processed & Saved graph for: {pdb_id}")

    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")

print(f"\nFeature extraction complete! Successfully processed {success_count}/30 graphs. Saved locally in data/processed/")
