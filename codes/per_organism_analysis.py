import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# ── Args from Snakemake wildcard ─────────────────────────────
organism  = sys.argv[1]   # e.g. "E.coli"
interface = sys.argv[2]   # e.g. "Human"
input_csv = sys.argv[3]   # combined_amr_data.csv
output_csv = sys.argv[4]  # results/E.coli_resistance_profile.csv
output_png = sys.argv[5]  # results/E.coli_resistance_plot.png

print(f"Processing: {organism} ({interface})")

# ── Load and filter ──────────────────────────────────────────
df = pd.read_csv(input_csv)
df_org = df[df['Organism'] == organism].copy()

print(f"Records for {organism}: {len(df_org)}")

# ── Resistance rate per antibiotic ───────────────────────────
df_resist = df_org[
    df_org['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])
].copy()

resist_rate = df_resist.groupby('Antibiotic').apply(
    lambda x: (x['Resistant Phenotype'] == 'Resistant'
              ).sum() / len(x)
).reset_index()
resist_rate.columns = ['Antibiotic', 'Resistance Rate']
resist_rate['Organism'] = organism
resist_rate['Interface'] = interface
resist_rate = resist_rate.sort_values(
    'Resistance Rate', ascending=False
)

# ── Save CSV ─────────────────────────────────────────────────
resist_rate.to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")

# ── Plot ─────────────────────────────────────────────────────
interface_colors = {
    'Human': '#e74c3c',
    'Animal': '#f39c12',
    'Environment': '#27ae60'
}

top20 = resist_rate.head(20)
plt.figure(figsize=(12, 6))
bars = plt.bar(
    top20['Antibiotic'],
    top20['Resistance Rate'],
    color=interface_colors.get(interface, '#3498db'),
    alpha=0.85, edgecolor='white', linewidth=0.5
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.ylabel('Resistance Rate', fontsize=11)
plt.title(
    f'{organism} — Top 20 Antibiotic Resistance Rates\n'
    f'Interface: {interface}',
    fontsize=13, fontweight='bold'
)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {output_png}")

print(f"Done: {organism}")