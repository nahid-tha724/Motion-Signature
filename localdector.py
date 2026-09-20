# Create necessary local directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Define an expanded benchmark list of 30 representative proteins
sample_pdbs = [
    "1ubq", "2lzm", "1ake", "3apo", "1cit",
    "4hhb", "1gcn", "1mbn", "2ci2", "1div",
    "1ejg", "1fkb", "1crn", "1hrc", "1izu",
    "1jfn", "1kte", "1l2y", "1mbs", "1nis",
    "2gb1", "1pga", "1sh1", "1bpi", "1aon",
    "1fna", "1gfl", "1hje", "1lmb", "1r69"
]

metadata_rows = []
for pdb_id in sample_pdbs:
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = f"data/raw/{pdb_id}.pdb"
    try:
        urlretrieve(url, out_path)
        metadata_rows.append({
            "protein_id": pdb_id,
            "sequence": "MKTIIAL...", # Placeholder sequence for demonstration
            "structure_path": out_path,
            "trajectory_path": "",
            "functional_label": hash(pdb_id) % 2, # Binary dummy label for testing pipeline functionality
            "family_id": f"Family_{pdb_id[0]}",
            "fold_id": f"Fold_{pdb_id[-1]}"
        })
        print(f"Downloaded: {pdb_id}")
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")

df_meta = pd.DataFrame(metadata_rows)
df_meta.to_csv("data/raw/metadata.csv", index=False)
print(f"Successfully curated {len(df_meta)} proteins for local pipeline run.")
