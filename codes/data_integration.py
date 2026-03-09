import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── File paths ───────────────────────────────────────────────
files = {
    'E.coli': {
        'path': r"F:\AMR_new_project\E.coli_BVBRC_genome_amr.csv",
        'interface': 'Human'
    },
    'S.enterica': {
        'path': r"F:\AMR_new_project\S.enterics_BVBRC_genome_amr.csv",
        'interface': 'Human'
    },
    'S.aureus': {
        'path': r"F:\AMR_new_project\Staphylococcus aureus BVBRC_genome_amr.csv",
        'interface': 'Animal'
    },
    'A.baumannii': {
        'path': r"F:\AMR_new_project\Acinetobacter baumannii BVBRC_genome_amr.csv",
        'interface': 'Environment'
    },
    'P.aeruginosa': {
        'path': r"F:\AMR_new_project\Pseudomonas aeruginosa BVBRC_genome_amr.csv",
        'interface': 'Environment'
    }
}

# ── Load and combine ─────────────────────────────────────────
dfs = []
for organism, info in files.items():
    df = pd.read_csv(info['path'])
    df['Organism'] = organism
    df['Interface'] = info['interface']
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

# ── Clean ────────────────────────────────────────────────────
combined = combined[[
    'Genome Name', 'Organism', 'Interface',
    'Antibiotic', 'Resistant Phenotype'
]].dropna()

combined['Resistant Phenotype'] = combined['Resistant Phenotype'].str.strip()

# ── Standardize phenotype ────────────────────────────────────
combined['Resistance Score'] = combined['Resistant Phenotype'].map({
    'Resistant': 1,
    'Nonsusceptible': 1,
    'Intermediate': 0.5,
    'Susceptible': 0
}).fillna(0)

# ── Save combined dataset ────────────────────────────────────
os.makedirs(r"F:\AMR_new_project\results", exist_ok=True)
combined.to_csv(
    r"F:\AMR_new_project\results\combined_amr_data.csv",
    index=False
)
print(f"Combined dataset: {combined.shape}")
print(f"Organisms: {combined['Organism'].unique()}")
print(f"Interfaces: {combined['Interface'].unique()}")
print(f"Total antibiotics: {combined['Antibiotic'].nunique()}")

# ── Plot 1: Resistance rate by interface ─────────────────────
interface_resist = combined.groupby(
    ['Interface', 'Resistant Phenotype']
).size().unstack(fill_value=0)

interface_resist.plot(
    kind='bar', figsize=(10, 6),
    color=['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4']
)
plt.title(
    'AMR Phenotype Distribution Across Human-Animal-Environment Interface\n'
    'One Health AMR-EvoNet Surveillance',
    fontsize=13, fontweight='bold'
)
plt.xlabel('Interface', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.xticks(rotation=0)
plt.legend(title='Phenotype')
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\interface_resistance_profile.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Interface resistance profile saved!")

# ── Plot 2: Resistance rate % per organism ───────────────────
resist_rate = combined[
    combined['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])
].groupby('Organism').apply(
    lambda x: (x['Resistant Phenotype'] == 'Resistant').sum() / len(x) * 100
).reset_index()
resist_rate.columns = ['Organism', 'Resistance Rate %']
resist_rate = resist_rate.sort_values('Resistance Rate %', ascending=False)

# Add interface color
interface_colors = {
    'E.coli': '#e74c3c',
    'S.enterica': '#c0392b',
    'S.aureus': '#f39c12',
    'A.baumannii': '#27ae60',
    'P.aeruginosa': '#2ecc71'
}
colors = [interface_colors[org] for org in resist_rate['Organism']]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    resist_rate['Organism'],
    resist_rate['Resistance Rate %'],
    color=colors
)
plt.title(
    'Resistance Rate by Organism\n'
    'Red=Human | Orange=Animal | Green=Environment',
    fontsize=13, fontweight='bold'
)
plt.xlabel('Organism', fontsize=11)
plt.ylabel('Resistance Rate (%)', fontsize=11)
plt.xticks(rotation=0)

# Add value labels on bars
for bar, val in zip(bars, resist_rate['Resistance Rate %']):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%',
        ha='center', fontsize=10
    )

plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\resistance_rate_by_organism.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Resistance rate plot saved!")

# ── Plot 3: Shared antibiotics heatmap across interfaces ──────
pivot = combined.groupby(
    ['Interface', 'Antibiotic']
)['Resistance Score'].mean().unstack(fill_value=0)

plt.figure(figsize=(20, 6))
sns.heatmap(
    pivot,
    cmap='Reds',
    vmin=0, vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Mean Resistance Score'}
)
plt.title(
    'Antibiotic Resistance Landscape Across Human-Animal-Environment Interface',
    fontsize=13, fontweight='bold'
)
plt.xlabel('Antibiotic', fontsize=11)
plt.ylabel('Interface', fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\interface_antibiotic_heatmap.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Interface heatmap saved!")

print("\nScript 1 complete! Check results folder.")
print(f"Saved: combined_amr_data.csv")
print(f"Saved: interface_resistance_profile.png")
print(f"Saved: resistance_rate_by_organism.png")
print(f"Saved: interface_antibiotic_heatmap.png")