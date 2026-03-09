import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import shap
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs(r"F:\AMR_new_project\results", exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(r"F:\AMR_new_project\results\combined_amr_data.csv")
network_metrics = pd.read_csv(
    r"F:\AMR_new_project\results\network_metrics.csv"
)

print(f"Loaded: {df.shape}")

# ── Feature Engineering ──────────────────────────────────────
# Keep only Resistant/Susceptible
df_ml = df[df['Resistant Phenotype'].isin(
    ['Resistant', 'Susceptible'])].copy()
df_ml['label'] = (
    df_ml['Resistant Phenotype'] == 'Resistant'
).astype(int)

# Encode categorical features
le_org = LabelEncoder()
le_ab = LabelEncoder()
le_interface = LabelEncoder()

df_ml['org_encoded'] = le_org.fit_transform(df_ml['Organism'])
df_ml['ab_encoded'] = le_ab.fit_transform(df_ml['Antibiotic'])
df_ml['interface_encoded'] = le_interface.fit_transform(
    df_ml['Interface']
)

# Add network centrality as feature
centrality_map = dict(zip(
    network_metrics['Antibiotic'],
    network_metrics['Degree Centrality']
))
df_ml['ab_centrality'] = df_ml['Antibiotic'].map(
    centrality_map
).fillna(0)

# Add interface resistance rate as feature
interface_rates = df_ml.groupby('Interface')['label'].mean()
df_ml['interface_resist_rate'] = df_ml['Interface'].map(interface_rates)

# Features
features = [
    'org_encoded',
    'ab_encoded',
    'interface_encoded',
    'ab_centrality',
    'interface_resist_rate'
]

X = df_ml[features]
y = df_ml['label']

print(f"Features: {features}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# ── Train/Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Random Forest ────────────────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"RF AUC: {rf_auc:.3f} | CV AUC: {rf_cv.mean():.3f} "
      f"± {rf_cv.std():.3f}")

# ── Gradient Boosting ────────────────────────────────────────
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)
gb.fit(X_train, y_train)
gb_prob = gb.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_prob)
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='roc_auc')
print(f"GB AUC: {gb_auc:.3f} | CV AUC: {gb_cv.mean():.3f} "
      f"± {gb_cv.std():.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"RF Stratified CV AUC: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

# ── Plot 1: ROC Curves ───────────────────────────────────────
plt.figure(figsize=(8, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)

plt.plot(fpr_rf, tpr_rf, color='#e74c3c', linewidth=2,
         label=f'Random Forest (AUC={rf_auc:.3f})')
plt.plot(fpr_gb, tpr_gb, color='#3498db', linewidth=2,
         label=f'Gradient Boosting (AUC={gb_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1,
         label='Random classifier')
plt.fill_between(fpr_rf, tpr_rf, alpha=0.1, color='#e74c3c')
plt.fill_between(fpr_gb, tpr_gb, alpha=0.1, color='#3498db')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(
    'ROC Curves — AMR Resistance Prediction\n'
    'Human-Animal-Environment Interface Model',
    fontsize=13, fontweight='bold'
)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\roc_curves.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("ROC curves saved!")

# ── Plot 2: Confusion Matrix ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, prob) in zip(
    axes,
    [('Random Forest', rf_prob), ('Gradient Boosting', gb_prob)]
):
    pred = (prob > 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                ax=ax, cbar=False,
                xticklabels=['Susceptible', 'Resistant'],
                yticklabels=['Susceptible', 'Resistant'])
    ax.set_title(f'{name} Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\confusion_matrix.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Confusion matrix saved!")

# ── Plot 3: SHAP Feature Importance ─────────────────────────
# Replace the entire SHAP section with this:
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

colors = ['#e74c3c', '#e67e22', '#f1c40f',
          '#2ecc71', '#3498db']
plt.barh(feature_importance['Feature'],
         feature_importance['Importance'],
         color=colors)
plt.xlabel('Feature Importance', fontsize=12)
plt.title(
    'Feature Importance — AMR Spillover Risk Model\n'
    'Key Drivers of Resistance Across Interfaces',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\shap_importance.png",
    dpi=300, bbox_inches='tight'
)
plt.show()

# ── Plot 4: Spillover Risk Score per Antibiotic ──────────────
# Novel output — risk score based on network centrality
# + interface resistance rate
print("Computing spillover risk scores...")

risk_df = network_metrics.copy()

# Add mean resistance rate per antibiotic across interfaces
ab_resist_rate = df_ml.groupby('Antibiotic')['label'].mean()
risk_df['Mean Resistance Rate'] = risk_df['Antibiotic'].map(
    ab_resist_rate
).fillna(0)

# Spillover Risk Score = centrality * resistance rate * betweenness
risk_df['Spillover Risk Score'] = (
    risk_df['Degree Centrality'] * 0.4 +
    risk_df['Betweenness Centrality'] * 0.3 +
    risk_df['Mean Resistance Rate'] * 0.3
)

risk_df = risk_df.sort_values(
    'Spillover Risk Score', ascending=False
)

# Plot top 15
top15 = risk_df.head(15)
colors = plt.cm.RdYlGn_r(
    np.linspace(0.2, 0.9, len(top15))
)

plt.figure(figsize=(12, 7))
bars = plt.barh(
    top15['Antibiotic'],
    top15['Spillover Risk Score'],
    color=colors
)
plt.xlabel('Spillover Risk Score', fontsize=12)
plt.title(
    'AMR Spillover Risk Score — Top 15 Antibiotics\n'
    'Based on Network Centrality + Cross-Interface Resistance Rate',
    fontsize=13, fontweight='bold'
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(
    r"F:\AMR_new_project\results\spillover_risk_scores.png",
    dpi=300, bbox_inches='tight'
)
plt.show()
print("Spillover risk scores saved!")

# Save results
risk_df.to_csv(
    r"F:\AMR_new_project\results\spillover_risk_scores.csv",
    index=False
)

print("\nScript 3 complete!")
print(f"RF AUC: {rf_auc:.3f}")
print(f"GB AUC: {gb_auc:.3f}")
print("\nTop 5 Spillover Risk Antibiotics:")
print(risk_df[['Antibiotic', 'Spillover Risk Score']].head())