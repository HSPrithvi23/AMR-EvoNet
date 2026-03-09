import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio import Phylo
from Bio.Phylo.TreeConstruction import (
    DistanceMatrix, DistanceTreeConstructor
)
import os

os.makedirs(r"F:\AMR_new_project\results", exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(r"F:\AMR_new_project\results\combined_amr_data.csv")
risk_scores = pd.read_csv(
    r"F:\AMR_new_project\results\spillover_risk_scores.csv"
)

print(f"Loaded: {df.shape}")

# ── Build AMR Profile Matrix per Organism ────────────────────
df['Is Resistant'] = df['Resistant Phenotype'].isin(
    ['Resistant', 'Nonsusceptible']
).astype(int)

# Get common antibiotics across all organisms
org_ab_matrix = df.groupby(
    ['Organism', 'Antibiotic']
)['Is Resistant'].mean().unstack(fill_value=0)

print(f"Organism-Antibiotic matrix: {org_ab_matrix.shape}")
print(f"Organisms: {list(org_ab_matrix.index)}")

# ── Compute Distance Matrix ──────────────────────────────────
from sklearn.metrics.pairwise import euclidean_distances

organisms = list(org_ab_matrix.index)
X = org_ab_matrix.values
dist_matrix = euclidean_distances(X)

print("\nDistance Matrix:")
dist_df = pd.DataFrame(
    dist_matrix,
    index=organisms,
    columns=organisms
)
print(dist_df.round(3))

# ── Build Phylogenetic Tree ──────────────────────────────────
# Convert to lower triangle for BioPython
n = len(organisms)
lower_triangle = []
for i in range(n):
    row = []
    for j in range(i + 1):
        row.append(float(dist_matrix[i][j]))
    lower_triangle.append(row)

dm = DistanceMatrix(organisms, lower_triangle)
constructor = DistanceTreeConstructor()
tree = constructor.nj(dm)

# ── Plot 1: Phylogenetic Tree colored by Interface ───────────
interface_map = {
    'E.coli': 'Human',
    'S.enterica': 'Human',
    'S.aureus': 'Animal',
    'A.baumannii': 'Environment',
    'P.aeruginosa': 'Environment'
}

interface_colors = {
    'Human': '#e74c3c',
    'Animal': '#f39c12',
    'Environment': '#27ae60'
}

fig, ax = plt.subplots(figsize=(12, 8))
Phylo.draw(tree, axes=ax, do_show=False,
           label_func=lambda x: x.name if x.name else '')

# Color the labels by interface
for text in ax.texts:
    label = text.get_text().strip()
    if label in interface_map:
        interface = interface_map[label]
        text.set_color(interface_colors[interface])
        text.set_fontsize(12)
        text.set_fontweight('bold')

ax.set_title(
    'Phylogenetic Tree — AMR Evolutionary Relationships\n'
    'Human-Animal-Environment Interface\n'
    '(Colored by interface: Red=Human, Orange=Animal, Green=Environment)',
    fontsize=13, fontweight='bold'
)

legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Human (E.coli, S.enterica)'),
    mpatches.Patch(color='#f39c12', label='Animal (S.aureus)'),
    mpatches.Patch(color='#27ae60',
                   label='Environment (A.baumannii, P.aeruginosa)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax.set_xlabel('Branch Length (AMR Profile Distance)', fontsize=11)

plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\phylogenetic_tree.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Phylogenetic tree saved!")

# ── Plot 2: AMR Profile Heatmap per Organism ─────────────────
# Show top 20 antibiotics by variance across organisms
top_ab = org_ab_matrix.var().sort_values(
    ascending=False
).head(20).index
matrix_top = org_ab_matrix[top_ab]

fig, ax = plt.subplots(figsize=(16, 6))
im = ax.imshow(
    matrix_top.values,
    cmap='RdYlGn_r',
    aspect='auto',
    vmin=0, vmax=1
)

ax.set_xticks(range(len(top_ab)))
ax.set_xticklabels(top_ab, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(organisms)))

# Color y labels by interface
yticklabels = ax.set_yticklabels(organisms, fontsize=11)
for label in yticklabels:
    org = label.get_text()
    if org in interface_map:
        label.set_color(
            interface_colors[interface_map[org]]
        )
        label.set_fontweight('bold')

plt.colorbar(im, ax=ax,
             label='Resistance Rate (0=Susceptible, 1=Resistant)')
ax.set_title(
    'AMR Resistance Profile Comparison Across Organisms\n'
    'Top 20 Most Variable Antibiotics',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\organism_amr_profiles.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Organism AMR profiles saved!")

# ── Plot 3: Evolutionary Distance Heatmap ────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(dist_matrix, cmap='Blues', aspect='auto')

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(organisms, rotation=45,
                   ha='right', fontsize=10)
ax.set_yticklabels(organisms, fontsize=10)

# Color labels
for i, label in enumerate(ax.get_xticklabels()):
    org = label.get_text()
    if org in interface_map:
        label.set_color(interface_colors[interface_map[org]])
        label.set_fontweight('bold')

for i, label in enumerate(ax.get_yticklabels()):
    org = label.get_text()
    if org in interface_map:
        label.set_color(interface_colors[interface_map[org]])
        label.set_fontweight('bold')

# Add distance values
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{dist_matrix[i,j]:.2f}',
                ha='center', va='center',
                fontsize=9, color='black')

plt.colorbar(im, ax=ax, label='Evolutionary Distance')
ax.set_title(
    'AMR Evolutionary Distance Matrix\n'
    'Based on Resistance Profile Similarity\n'
    'Red=Human | Orange=Animal | Green=Environment',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\evolutionary_distance_matrix.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Evolutionary distance matrix saved!")

# ── Save distance data ───────────────────────────────────────
dist_df.to_csv(
    r"F:\AMR_new_project\results\evolutionary_distances.csv"
)

print("\nScript 4 complete!")
print("Saved: phylogenetic_tree.png")
print("Saved: organism_amr_profiles.png")
print("Saved: evolutionary_distance_matrix.png")
print("\nKey Finding:")
print("Shortest distance = most similar AMR profiles = "
      "highest gene transfer risk")
print(dist_df.round(3))