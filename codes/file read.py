import pandas as pd

files = {
    'E.coli (Human)': r"F:\AMR_new_project\E.coli_BVBRC_genome_amr.csv",
    'S.enterica (Human)': r"F:\AMR_new_project\S.enterics_BVBRC_genome_amr.csv",
    'Staphylococcus (Animal)': r"F:\AMR_new_project\Staphylococcus aureus BVBRC_genome_amr.csv",
    'Acinetobacter (Environment)': r"F:\AMR_new_project\Acinetobacter baumannii BVBRC_genome_amr.csv",
    'Pseudomonas (Environment)': r"F:\AMR_new_project\Pseudomonas aeruginosa BVBRC_genome_amr.csv",
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)
        print(f"{name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Genomes: {df['Genome Name'].nunique()}")
        print(f"  Antibiotics: {df['Antibiotic'].nunique()}")
        print(f"  Resistant: {(df['Resistant Phenotype']=='Resistant').sum()}")
        print()
    except Exception as e:
        print(f"{name}: ERROR — {e}")