import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from itertools import combinations
import os

# ── Load combined data ───────────────────────────────────────
df = pd.read_csv(r"F:\AMR_new_project\results\combined_amr_data.csv")

os.makedirs(r"F:\AMR_new_project\results", exist_ok=True)

# ── Build antibiotic resistance matrix per organism ──────────
df['Is Resistant'] = df['Resistant Phenotype'].isin(
    ['Resistant', 'Nonsusceptible']
).astype(int)

# ── Network 1: Antibiotic Co-resistance Network ──────────────
# Which antibiotics are co-resisted across all organisms?
print("Building co-resistance network...")

pivot = df.pivot_table(
    index='Genome Name',
    columns='Antibiotic',
    values='Is Resistant',
    aggfunc='max'
).fillna(0)

# Build co-resistance edges
G_coR = nx.Graph()
min_co = 50  # minimum co-resistance count

for ab1, ab2 in combinations(pivot.columns, 2):
    co = ((pivot[ab1] == 1) & (pivot[ab2] == 1)).sum()
    if co >= min_co:
        G_coR.add_edge(ab1, ab2, weight=int(co))

print(f"Co-resistance network: {G_coR.number_of_nodes()} nodes, "
      f"{G_coR.number_of_edges()} edges")

# ── Network 2: Interface Gene Flow Network ───────────────────
# Which antibiotics are shared across Human-Animal-Environment?
print("Building interface gene flow network...")

# Resistance rate per antibiotic per interface
interface_resist = df.groupby(
    ['Interface', 'Antibiotic']
)['Is Resistant'].mean().reset_index()
interface_resist.columns = ['Interface', 'Antibiotic', 'Resistance Rate']

# Find antibiotics resistant in ALL three interfaces
interfaces = ['Human', 'Animal', 'Environment']
shared_antibiotics = {}

for ab in df['Antibiotic'].unique():
    ab_data = interface_resist[interface_resist['Antibiotic'] == ab]
    present_in = ab_data[
        ab_data['Resistance Rate'] > 0.1
    ]['Interface'].tolist()
    if len(present_in) >= 2:
        shared_antibiotics[ab] = present_in

print(f"Antibiotics shared across 2+ interfaces: {len(shared_antibiotics)}")

# Build interface flow network
G_flow = nx.Graph()

# Add interface nodes
interface_colors = {
    'Human': '#e74c3c',
    'Animal': '#f39c12',
    'Environment': '#27ae60'
}

for interface in interfaces:
    G_flow.add_node(interface,
                   node_type='interface',
                   color=interface_colors[interface])

# Add antibiotic nodes and edges
for ab, present_in in shared_antibiotics.items():
    if len(present_in) >= 2:
        G_flow.add_node(ab, node_type='antibiotic', color='#3498db')
        for interface in present_in:
            rate = interface_resist[
                (interface_resist['Interface'] == interface) &
                (interface_resist['Antibiotic'] == ab)
            ]['Resistance Rate'].values
            weight = float(rate[0]) if len(rate) > 0 else 0.1
            G_flow.add_edge(interface, ab, weight=weight)

print(f"Gene flow network: {G_flow.number_of_nodes()} nodes, "
      f"{G_flow.number_of_edges()} edges")

# ── Plot 1: Co-resistance Network ────────────────────────────
plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G_coR, seed=42, k=2)

degree_cent = nx.degree_centrality(G_coR)
node_size = [500 + 3000 * degree_cent.get(n, 0)
             for n in G_coR.nodes()]
node_color = [degree_cent.get(n, 0) for n in G_coR.nodes()]

edge_weights = [G_coR[u][v]['weight'] for u, v in G_coR.edges()]
max_w = max(edge_weights) if edge_weights else 1
edge_width = [0.5 + 3 * w/max_w for w in edge_weights]

nx.draw_networkx_edges(G_coR, pos,
                       width=edge_width,
                       alpha=0.3,
                       edge_color='grey')
nodes = nx.draw_networkx_nodes(G_coR, pos,
                                node_size=node_size,
                                node_color=node_color,
                                cmap=plt.cm.RdYlGn_r,
                                alpha=0.9)
nx.draw_networkx_labels(G_coR, pos,
                        font_size=7,
                        font_color='black')

plt.colorbar(nodes, label='Degree Centrality')
plt.title(
    'AMR Co-Resistance Network\n'
    'Node size = connectivity | Color = centrality',
    fontsize=14, fontweight='bold'
)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\coresistance_network.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Co-resistance network saved!")

# ── Plot 2: Interface Gene Flow Network ──────────────────────
plt.figure(figsize=(20, 16))

# Separate interface and antibiotic nodes
interface_nodes = [n for n, d in G_flow.nodes(data=True)
                   if d.get('node_type') == 'interface']
ab_nodes = [n for n, d in G_flow.nodes(data=True)
            if d.get('node_type') == 'antibiotic']

# Position interface nodes as triangle
pos_flow = {}
pos_flow['Human'] = np.array([0.5, 0.9])
pos_flow['Animal'] = np.array([0.1, 0.1])
pos_flow['Environment'] = np.array([0.9, 0.1])

# Position antibiotic nodes using spring layout
G_ab_only = G_flow.subgraph(ab_nodes)
pos_ab = nx.spring_layout(G_ab_only, seed=42, k=0.3)

# Scale antibiotic positions to middle area
for node, p in pos_ab.items():
    pos_flow[node] = np.array([
        0.2 + 0.6 * (p[0] + 1) / 2,
        0.2 + 0.6 * (p[1] + 1) / 2
    ])

# Draw edges
edge_colors = []
for u, v in G_flow.edges():
    if u in interface_nodes or v in interface_nodes:
        interface = u if u in interface_nodes else v
        edge_colors.append(interface_colors.get(interface, 'grey'))
    else:
        edge_colors.append('lightgrey')

nx.draw_networkx_edges(
    G_flow, pos_flow,
    edge_color=edge_colors,
    alpha=0.4, width=1.5
)

# Draw interface nodes (large)
nx.draw_networkx_nodes(
    G_flow, pos_flow,
    nodelist=interface_nodes,
    node_color=[interface_colors[n] for n in interface_nodes],
    node_size=3000,
    alpha=0.9
)

# Draw antibiotic nodes (small, blue)
nx.draw_networkx_nodes(
    G_flow, pos_flow,
    nodelist=ab_nodes,
    node_color='#3498db',
    node_size=300,
    alpha=0.6
)

# Labels for interface nodes
nx.draw_networkx_labels(
    G_flow, pos_flow,
    labels={n: n for n in interface_nodes},
    font_size=14, font_weight='bold',
    font_color='white'
)

# Labels for antibiotic nodes
nx.draw_networkx_labels(
    G_flow, pos_flow,
    labels={n: n for n in ab_nodes},
    font_size=6,
    font_color='black'
)

# Legend
legend_elements = [
    mpatches.Patch(color='#e74c3c', label='Human Interface'),
    mpatches.Patch(color='#f39c12', label='Animal Interface'),
    mpatches.Patch(color='#27ae60', label='Environment Interface'),
    mpatches.Patch(color='#3498db', label='Shared Antibiotic Resistance')
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.title(
    'AMR Gene Flow Network — Human-Animal-Environment Interface\n'
    'Blue nodes = antibiotics with shared resistance across interfaces\n'
    'Edges = resistance connection between interface and antibiotic',
    fontsize=13, fontweight='bold'
)
plt.axis('off')
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\interface_geneflow_network.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Gene flow network saved!")

# ── Save network metrics ─────────────────────────────────────
degree_cent = nx.degree_centrality(G_coR)
betweenness = nx.betweenness_centrality(G_coR)
closeness = nx.closeness_centrality(G_coR)

metrics_df = pd.DataFrame({
    'Antibiotic': list(degree_cent.keys()),
    'Degree Centrality': list(degree_cent.values()),
    'Betweenness Centrality': [betweenness[n]
                               for n in degree_cent.keys()],
    'Closeness Centrality': [closeness[n]
                             for n in degree_cent.keys()]
}).sort_values('Degree Centrality', ascending=False)

metrics_df.to_csv(
    r"F:\AMR_new_project\results\network_metrics.csv",
    index=False
)

print("\nTop 10 AMR Hubs:")
print(metrics_df.head(10).to_string(index=False))

print("\nScript 2 complete!")
print("Saved: coresistance_network.png")
print("Saved: interface_geneflow_network.png")
print("Saved: network_metrics.csv")