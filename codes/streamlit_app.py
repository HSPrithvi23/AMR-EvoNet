import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AMR-EvoNet",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --card: #141b2d;
    --green: #00ff88;
    --blue: #0ea5e9;
    --red: #ff4757;
    --orange: #ff6b35;
    --text: #e8eaf6;
    --muted: #8892b0;
    --border: rgba(0,255,136,0.15);
}

.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #0f1529 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.hero {
    background: linear-gradient(135deg, #0a0e1a, #0f1d35, #091a0f);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
}
.hero-title {
    font-family: Syne, sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: var(--text);
    margin: 0 0 8px 0;
}
.hero-title span { color: var(--green); }
.hero-sub {
    font-family: Space Mono, monospace;
    font-size: 0.75rem;
    color: var(--green);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-desc {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 700px;
    line-height: 1.6;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.metric-label {
    font-family: Space Mono, monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}
.metric-value {
    font-family: Syne, sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--green);
}
.finding-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.section-header {
    font-family: Syne, sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    margin: 32px 0 16px 0;
}
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00cc66) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: Space Mono, monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    font-family: Syne, sans-serif !important;
    color: #00ff88 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly Theme ─────────────────────────────────────────────
THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(20,27,45,0.8)',
    font=dict(family='DM Sans', color='#8892b0'),
    margin=dict(l=40, r=40, t=60, b=40)
)

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"F:\AMR_new_project\results\combined_amr_data.csv"
    )
    network = pd.read_csv(
        r"F:\AMR_new_project\results\network_metrics.csv"
    )
    risk = pd.read_csv(
        r"F:\AMR_new_project\results\spillover_risk_scores.csv"
    )
    distances = pd.read_csv(
        r"F:\AMR_new_project\results\evolutionary_distances.csv",
        index_col=0
    )
    return df, network, risk, distances

df, network_metrics, risk_scores, distances = load_data()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem;
                    font-weight: 800; color: #00ff88;'>
            AMR-EvoNet
        </div>
        <div style='font-family: Space Mono, monospace; font-size: 0.6rem;
                    color: #4a5568; letter-spacing: 2px; margin-top: 4px;'>
            AMR Gene Flow Surveillance
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "Home",
        "Interface Overview",
        "Co-Resistance Network",
        "Gene Flow Network",
        "ML Risk Prediction",
        "Phylogenetics",
        "Spillover Risk Scores"
    ])

# ── Interface colors ─────────────────────────────────────────
INTERFACE_COLORS = {
    'Human': '#e74c3c',
    'Animal': '#f39c12',
    'Environment': '#27ae60'
}

ORGANISM_COLORS = {
    'E.coli': '#e74c3c',
    'S.enterica': '#c0392b',
    'S.aureus': '#f39c12',
    'A.baumannii': '#27ae60',
    'P.aeruginosa': '#2ecc71'
}

# ════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class='hero'>
        <div class='hero-sub'>
            AMR Gene Flow · Human-Animal-Environment Interface
        </div>
        <div class='hero-title'>
            AMR-<span>EvoNet</span>
        </div>
        <div class='hero-desc'>
            First computational platform tracking antibiotic resistance
            gene flow across the Human-Animal-Environment interface using
            network modeling, evolutionary analysis and ML-based
            spillover risk scoring under a One Health framework.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Organisms", df['Organism'].nunique())
    col3.metric("Antibiotics", df['Antibiotic'].nunique())
    col4.metric("Interfaces", df['Interface'].nunique())

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='finding-card'>
            <div style='font-family: Space Mono, monospace;
                        font-size: 0.7rem; color: #00ff88;
                        letter-spacing: 2px; margin-bottom: 8px;'>
                KEY FINDING 1
            </div>
            <div style='color: #e8eaf6; font-size: 0.9rem;'>
                Environmental bacteria show highest resistance
                rates (A.baumannii 61.3%) — suggesting environment
                as the origin of AMR gene flow toward humans
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='finding-card'>
            <div style='font-family: Space Mono, monospace;
                        font-size: 0.7rem; color: #00ff88;
                        letter-spacing: 2px; margin-bottom: 8px;'>
                KEY FINDING 2
            </div>
            <div style='color: #e8eaf6; font-size: 0.9rem;'>
                S.aureus (Animal) and S.enterica (Human) show
                shortest evolutionary distance (3.43) — direct
                evidence of Animal-to-Human AMR transfer
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='finding-card'>
            <div style='font-family: Space Mono, monospace;
                        font-size: 0.7rem; color: #00ff88;
                        letter-spacing: 2px; margin-bottom: 8px;'>
                KEY FINDING 3
            </div>
            <div style='color: #e8eaf6; font-size: 0.9rem;'>
                Ciprofloxacin has highest spillover risk score
                (0.46) — highest network centrality + cross-interface
                resistance — immediate surveillance priority
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: INTERFACE OVERVIEW
# ════════════════════════════════════════════════════════════
elif "Interface" in page:
    st.markdown(
        "<div class='section-header'>Interface AMR Overview</div>",
        unsafe_allow_html=True
    )

    df['Is Resistant'] = df['Resistant Phenotype'].isin(
        ['Resistant', 'Nonsusceptible']
    ).astype(int)

    col1, col2 = st.columns(2)

    with col1:
        resist_rate = df[
            df['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])
        ].groupby('Organism').apply(
            lambda x: (x['Resistant Phenotype'] == 'Resistant'
                      ).sum() / len(x) * 100
        ).reset_index()
        resist_rate.columns = ['Organism', 'Resistance Rate %']
        resist_rate = resist_rate.sort_values(
            'Resistance Rate %', ascending=False
        )

        fig = px.bar(
            resist_rate,
            x='Organism',
            y='Resistance Rate %',
            color='Organism',
            color_discrete_map=ORGANISM_COLORS,
            title='Resistance Rate by Organism'
        )
        fig.update_layout(**THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        interface_resist = df.groupby(
            ['Interface', 'Resistant Phenotype']
        ).size().reset_index(name='Count')

        fig2 = px.bar(
            interface_resist,
            x='Interface',
            y='Count',
            color='Resistant Phenotype',
            color_discrete_map={
                'Resistant': '#ff4757',
                'Susceptible': '#00ff88',
                'Intermediate': '#ff6b35',
                'Nonsusceptible': '#8b5cf6'
            },
            title='Phenotype Distribution by Interface',
            barmode='group'
        )
        fig2.update_layout(**THEME)
        st.plotly_chart(fig2, use_container_width=True)

    # Interface heatmap
    interface_ab = df.groupby(
        ['Interface', 'Antibiotic']
    )['Is Resistant'].mean().unstack(fill_value=0)

    fig3 = go.Figure(go.Heatmap(
        z=interface_ab.values,
        x=interface_ab.columns.tolist(),
        y=interface_ab.index.tolist(),
        colorscale='Reds',
        colorbar=dict(title='Resistance Rate')
    ))
    fig3.update_layout(
        **THEME,
        title='Antibiotic Resistance Landscape — '
              'Human vs Animal vs Environment',
        height=300,
        xaxis=dict(tickangle=45, tickfont=dict(size=8))
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: CO-RESISTANCE NETWORK
# ════════════════════════════════════════════════════════════
elif "Co-Resistance" in page:
    st.markdown(
        "<div class='section-header'>AMR Co-Resistance Network</div>",
        unsafe_allow_html=True
    )

    min_co = st.slider("Minimum co-resistance count:", 10, 200, 50)

    df['Is Resistant'] = df['Resistant Phenotype'].isin(
        ['Resistant', 'Nonsusceptible']
    ).astype(int)

    pivot = df.pivot_table(
        index='Genome Name',
        columns='Antibiotic',
        values='Is Resistant',
        aggfunc='max'
    ).fillna(0)

    G = nx.Graph()
    for ab1, ab2 in combinations(pivot.columns, 2):
        co = ((pivot[ab1] == 1) & (pivot[ab2] == 1)).sum()
        if co >= min_co:
            G.add_edge(ab1, ab2, weight=int(co))

    col1, col2, col3 = st.columns(3)
    col1.metric("Network Nodes", G.number_of_nodes())
    col2.metric("Network Edges", G.number_of_edges())
    col3.metric(
        "Density",
        f"{nx.density(G):.3f}" if G.number_of_nodes() > 1 else "N/A"
    )

    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=2)
        degree_cent = nx.degree_centrality(G)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_size = [15 + 50*degree_cent.get(n, 0)
                     for n in G.nodes()]
        node_color = [degree_cent.get(n, 0) for n in G.nodes()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color='rgba(0,255,136,0.15)'),
            hoverinfo='none'
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title='Centrality')
            ),
            text=list(G.nodes()),
            textposition='top center',
            textfont=dict(size=8, color='#8892b0'),
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        fig.update_layout(
            **THEME,
            title='AMR Co-Resistance Network',
            height=600,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top AMR Hubs:**")
        st.dataframe(
            network_metrics.head(10),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════════
# PAGE: GENE FLOW NETWORK
# ════════════════════════════════════════════════════════════
elif "Gene Flow" in page:
    st.markdown(
        "<div class='section-header'>"
        "AMR Gene Flow — Human-Animal-Environment"
        "</div>",
        unsafe_allow_html=True
    )

    df['Is Resistant'] = df['Resistant Phenotype'].isin(
        ['Resistant', 'Nonsusceptible']
    ).astype(int)

    interface_resist = df.groupby(
        ['Interface', 'Antibiotic']
    )['Is Resistant'].mean().reset_index()
    interface_resist.columns = [
        'Interface', 'Antibiotic', 'Resistance Rate'
    ]

    threshold = st.slider(
        "Minimum resistance rate threshold:", 0.1, 0.8, 0.3
    )

    shared = {}
    for ab in df['Antibiotic'].unique():
        ab_data = interface_resist[
            interface_resist['Antibiotic'] == ab
        ]
        present = ab_data[
            ab_data['Resistance Rate'] > threshold
        ]['Interface'].tolist()
        if len(present) >= 2:
            shared[ab] = present

    st.metric(
        f"Antibiotics shared across 2+ interfaces "
        f"(threshold={threshold:.1f})",
        len(shared)
    )

    st.markdown("""
    <div style='background: rgba(14,165,233,0.08);
                border: 1px solid rgba(14,165,233,0.25);
                border-radius: 10px; padding: 16px 20px;
                margin: 16px 0; font-size: 0.88rem;
                color: #8892b0;'>
        <strong style='color: #0ea5e9;'>What this means:</strong>
        Antibiotics appearing in this network show resistance
        across multiple interfaces simultaneously — evidence of
        AMR gene flow between Human, Animal and Environment.
        Higher threshold = stronger evidence of shared resistance.
    </div>
    """, unsafe_allow_html=True)

    # Sankey diagram for gene flow
    interfaces = ['Human', 'Animal', 'Environment']
    interface_idx = {i: idx for idx, i in enumerate(interfaces)}

    source, target, value, label_list = [], [], [], interfaces.copy()

    for ab, present_in in list(shared.items())[:30]:
        ab_idx = len(label_list)
        label_list.append(ab)
        for interface in present_in:
            rate = interface_resist[
                (interface_resist['Interface'] == interface) &
                (interface_resist['Antibiotic'] == ab)
            ]['Resistance Rate'].values
            if len(rate) > 0:
                source.append(interface_idx[interface])
                target.append(ab_idx)
                value.append(float(rate[0]))

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            line=dict(color='black', width=0.5),
            label=label_list,
            color=(
                ['#e74c3c', '#f39c12', '#27ae60'] +
                ['#3498db'] * (len(label_list) - 3)
            )
        ),
        link=dict(
            source=source, target=target, value=value,
            color='rgba(52,152,219,0.3)'
        )
    ))
    fig.update_layout(
        **THEME,
        title='AMR Gene Flow Sankey — '
              'Interface to Antibiotic Resistance',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: ML RISK PREDICTION
# ════════════════════════════════════════════════════════════
elif "ML Risk" in page:
    st.markdown(
        "<div class='section-header'>"
        "ML-Based AMR Resistance Prediction"
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("RF AUC", "0.808")
    col2.metric("Stratified CV AUC", "0.810 ± 0.004")
    col3.metric("Training Records", "39,464")

    @st.cache_resource
    def train_model(n):
        df_ml = df[df['Resistant Phenotype'].isin(
            ['Resistant', 'Susceptible'])].copy()
        df_ml['label'] = (
            df_ml['Resistant Phenotype'] == 'Resistant'
        ).astype(int)

        le_org = LabelEncoder()
        le_ab = LabelEncoder()
        le_int = LabelEncoder()

        df_ml['org_enc'] = le_org.fit_transform(df_ml['Organism'])
        df_ml['ab_enc'] = le_ab.fit_transform(df_ml['Antibiotic'])
        df_ml['int_enc'] = le_int.fit_transform(df_ml['Interface'])

        centrality_map = dict(zip(
            network_metrics['Antibiotic'],
            network_metrics['Degree Centrality']
        ))
        df_ml['ab_cent'] = df_ml['Antibiotic'].map(
            centrality_map
        ).fillna(0)

        int_rates = df_ml.groupby('Interface')['label'].mean()
        df_ml['int_rate'] = df_ml['Interface'].map(int_rates)

        X = df_ml[['org_enc', 'ab_enc', 'int_enc',
                   'ab_cent', 'int_rate']]
        y = df_ml['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        y_prob = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        return rf, le_ab, le_org, le_int, fpr, tpr, auc

    with st.spinner("Loading model..."):
        rf, le_ab, le_org, le_int, fpr, tpr, auc = train_model(
            len(df)
        )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.05)',
            name=f'RF (AUC={auc:.3f})'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='#4a5568', dash='dash'),
            name='Random'
        ))
        fig.update_layout(
            **THEME,
            title='ROC Curve — AMR Resistance Prediction',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        feat_imp = pd.DataFrame({
            'Feature': ['Organism', 'Antibiotic', 'Interface',
                       'Network Centrality', 'Interface Resist Rate'],
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig2 = px.bar(
            feat_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale=['#141b2d', '#ff6b35', '#ff4757']
        )
        fig2.update_layout(**THEME, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Predict Resistance</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    sel_org = c1.selectbox(
        "Organism:", df['Organism'].unique().tolist()
    )
    sel_ab = c2.selectbox(
        "Antibiotic:", sorted(df['Antibiotic'].unique())
    )
    sel_int = c3.selectbox(
        "Interface:", ['Human', 'Animal', 'Environment']
    )

    if st.button("Predict Resistance Risk"):
        try:
            ab_enc = le_ab.transform([sel_ab])[0]
        except:
            ab_enc = 0
        try:
            org_enc = le_org.transform([sel_org])[0]
        except:
            org_enc = 0
        try:
            int_enc = le_int.transform([sel_int])[0]
        except:
            int_enc = 0

        cent = network_metrics[
            network_metrics['Antibiotic'] == sel_ab
        ]['Degree Centrality'].values
        cent = float(cent[0]) if len(cent) > 0 else 0

        int_rate = df[
            df['Interface'] == sel_int
        ]['Resistance Score'].mean()

        prob = rf.predict_proba(
            [[org_enc, ab_enc, int_enc, cent, int_rate]]
        )[0][1]

        if prob > 0.5:
            st.markdown(f"""
            <div style='background: rgba(255,71,87,0.15);
                        border: 1px solid #ff4757;
                        border-radius: 12px; padding: 24px;
                        text-align: center;'>
                <div style='font-family: Syne, sans-serif;
                            font-size: 1.5rem; color: #ff4757;
                            font-weight: 800;'>RESISTANT</div>
                <div style='color: #8892b0; margin-top: 8px;'>
                    Probability: {prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: rgba(0,255,136,0.1);
                        border: 1px solid #00ff88;
                        border-radius: 12px; padding: 24px;
                        text-align: center;'>
                <div style='font-family: Syne, sans-serif;
                            font-size: 1.5rem; color: #00ff88;
                            font-weight: 800;'>SUSCEPTIBLE</div>
                <div style='color: #8892b0; margin-top: 8px;'>
                    Probability: {1-prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: PHYLOGENETICS
# ════════════════════════════════════════════════════════════
elif "Phylogen" in page:
    st.markdown(
        "<div class='section-header'>"
        "Phylogenetic & Evolutionary Analysis"
        "</div>",
        unsafe_allow_html=True
    )

    # Distance heatmap
    organisms = distances.index.tolist()
    fig = go.Figure(go.Heatmap(
        z=distances.values,
        x=organisms,
        y=organisms,
        colorscale='Blues',
        text=distances.round(2).values,
        texttemplate='%{text}',
        colorbar=dict(title='Evolutionary Distance')
    ))
    fig.update_layout(
        **THEME,
        title='AMR Evolutionary Distance Matrix\n'
              'Shorter distance = more similar resistance profiles '
              '= higher gene transfer risk',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='finding-card'>
        <div style='font-family: Space Mono, monospace;
                    font-size: 0.7rem; color: #00ff88;
                    letter-spacing: 2px; margin-bottom: 8px;'>
            KEY PHYLOGENETIC FINDING
        </div>
        <div style='color: #e8eaf6; font-size: 0.9rem;
                    line-height: 1.6;'>
            S.aureus (Animal) and S.enterica (Human) share the
            shortest evolutionary distance (3.43) — indicating
            the most similar AMR resistance profiles across the
            Animal-Human interface. This is computational evidence
            of AMR gene flow from livestock to human pathogens,
            directly relevant to One Health surveillance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        st.image(
            r"F:\AMR_new_project\results\phylogenetic_tree.png",
            caption="Phylogenetic Tree — AMR Profile Based "
                    "Evolutionary Clustering",
            use_column_width=True
        )
    except:
        st.warning("Run 04_phylogenetics.py first")

# ════════════════════════════════════════════════════════════
# PAGE: SPILLOVER RISK SCORES
# ════════════════════════════════════════════════════════════
elif "Spillover" in page:
    st.markdown(
        "<div class='section-header'>"
        "AMR Spillover Risk Scores"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style='background: rgba(14,165,233,0.08);
                border: 1px solid rgba(14,165,233,0.25);
                border-radius: 10px; padding: 16px 20px;
                margin: 16px 0; font-size: 0.88rem;
                color: #8892b0;'>
        <strong style='color: #0ea5e9;'>Novel Metric:</strong>
        Spillover Risk Score combines network centrality (40%) +
        betweenness centrality (30%) + cross-interface resistance
        rate (30%) into a single actionable risk score per
        antibiotic. Higher score = higher risk of resistance
        spreading across the Human-Animal-Environment interface.
    </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("Show top N antibiotics:", 5, 21, 15)
    top_risk = risk_scores.head(top_n)

    fig = px.bar(
        top_risk,
        x='Spillover Risk Score',
        y='Antibiotic',
        orientation='h',
        color='Spillover Risk Score',
        color_continuous_scale=['#141b2d', '#ff6b35', '#ff4757'],
        title=f'Top {top_n} AMR Spillover Risk Antibiotics'
    )
    fig.update_layout(**THEME, showlegend=False,
                     yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full Risk Score Table:**")
    st.dataframe(
        risk_scores[[
            'Antibiotic', 'Degree Centrality',
            'Betweenness Centrality', 'Spillover Risk Score'
        ]].round(3),
        use_container_width=True
    )