# AMR-EvoNet
One Health AMR Gene Flow Surveillance Platform — Human-Animal-Environment Interface

# AMR-EvoNet: One Health AMR Gene Flow Surveillance Pipeline - First computational platform tracking antibiotic resistance gene flow across the Human-Animal-Environment interface using network modeling, evolutionary analysis and ML-based spillover risk scoring under a One Health framework.

## Project Overview

AMR-EvoNet is an integrated bioinformatics surveillance pipeline designed to track the flow of antimicrobial resistance (AMR) genes across the **Human-Animal-Environment (HAE) interface** - the three interconnected domains of the One Health framework. This project was developed in direct relevance to the **NLM AMR Project**: "Epidemiological Surveillance of Antimicrobial Use and Resistance in Sheep, Goats and Poultry with One Health Approach.

### Research Questions Addressed
1. Which organisms act as AMR reservoirs at each Human-Animal-Environment interface?
2. Which antibiotics show cross-interface co-resistance — indicating gene flow?
3. Can ML models predict resistance phenotype using network centrality features?
4. Which antibiotics carry the highest spillover risk from animal/environment to human?

## 🌍 One Health Framework

```
        HUMAN Interface
        (E.coli, S.enterica)
              /\
             /  \
            /    \
           /      \
    ANIMAL -------- ENVIRONMENT
    Interface        Interface
    (S.aureus)    (A.baumannii,
                   P.aeruginosa)
```

AMR genes flow bidirectionally across these interfaces through:
- Direct animal-human contact
- Food chain contamination
- Environmental reservoirs (soil, water, hospital surfaces)
- Horizontal gene transfer between organisms
- 
## Dataset
### Organisms Analyzed

| Organism | Interface | Records | Source |
| *Escherichia coli* | Human | ~9,900 | BV-BRC |
| *Salmonella enterica* | Human | ~9,900 | BV-BRC |
| *Staphylococcus aureus* | Animal | ~9,900 | BV-BRC |
| *Acinetobacter baumannii* | Environment | ~9,900 | BV-BRC |
| *Pseudomonas aeruginosa* | Environment | ~9,900 | BV-BRC |

**Total: ~49,474 AMR phenotype records across 103 antibiotics**

### Data Sources
1] **BV-BRC (Bacterial and Viral Bioinformatics Resource Center)**: AMR phenotype data (Resistant/Susceptible/Intermediate classifications per genome per antibiotic)
  - URL: https://www.bv-brc.org/
  - Download path: Organism → AMR Phenotypes → Download CSV
  - Columns used: `Genome Name`, `Antibiotic`, `Resistant Phenotype`, `Measurement Value`, `Laboratory Typing Method`

2] **NCBI Genome Datasets**: Complete genome FASTA sequences for phylogenetic analysis
  - URL: https://www.ncbi.nlm.nih.gov/datasets/genome/
  - Filters: Assembly level = Complete, NCBI RefSeq annotated, Year ≥ 2015
  - Format: `.fna` (genome sequence) + `.gff` (annotation)

3] **CARD (Comprehensive Antibiotic Resistance Database)**: Reference database for resistance gene ontology
  - URL: https://card.mcmaster.ca/download
  - Files: `card-data.tar.bz2`, `card-ontology.tar.bz2`

> **Note:** Raw CSV data files are not included in this repository due to size. Download instructions are provided above.

---

## Features

### Snakemake Pipeline
- Config-driven workflow (`config.yaml`)
- Reproducible, modular pipeline with DAG visualization
- Parallel execution of per-organism analyses

### Data Integration (`data_integration.py`)
- Loads and standardizes AMR data from 5 organisms
- Assigns Human-Animal-Environment interface labels
- Generates combined resistance matrix (49,474 records × 103 antibiotics)
- Outputs: phenotype distribution plots, interface-level heatmaps

### Per-Organism Analysis (`per_organism_analysis.py`)
- Top 20 antibiotic resistance rates per organism
- Interface-colored visualizations (Red=Human, Orange=Animal, Green=Environment)
- Cross-organism resistance profile comparison heatmap

### Network Analysis (`network_analysis.py`)
- AMR co-resistance network (antibiotics as nodes, co-resistance as edges)
- AMR Gene Flow Network across Human-Animal-Environment interfaces
- Network centrality metrics: degree, betweenness, eigenvector
- Identifies AMR hub antibiotics driving multi-drug resistance

### ML Risk Scoring (`ml_risk_scoring.py`)
- Random Forest classifier (AUC = 0.808)
- Gradient Boosting classifier (AUC = 0.791)
- Features: antibiotic identity, organism, interface, network centrality, interface resistance rate
- SHAP feature importance — `ab_encoded` and `interface_resist_rate` are top drivers
- AMR Spillover Risk Scores combining network centrality + cross-interface resistance

### Phylogenetics (`phylogenetics.py`)
- NJ tree construction using AMR profile similarity (resistance-based distance matrix)
- Evolutionary distance matrix between all 5 organisms
- S.aureus–S.enterica closest pair (distance = 3.43) → Animal-to-Human AMR transfer risk

### Streamlit App (`streamlit_app.py`)
- Interactive AMR-EvoNet dashboard
- 7 pages: Home, Interface Overview, Co-Resistance Network, Gene Flow Network, ML Risk Prediction, Phylogenetics, Spillover Risk Scores
- Universal CSV upload — works with any AMR dataset format

## 🔬 Pipeline Architecture

```
config.yaml
    │
    ▼
data_integration.py
    ├── Combined dataset (49,474 records)
    ├── Interface assignment
    └── Resistance matrix
         │
         ├──────────────────────────┐
         ▼                          ▼
per_organism_analysis.py    network_analysis.py
    ├── E.coli                  ├── Co-resistance network
    ├── S.enterica              ├── Gene flow network
    ├── S.aureus                └── Centrality metrics
    ├── A.baumannii                      │
    └── P.aeruginosa                     ▼
                                ml_risk_scoring.py
                                    ├── Random Forest
                                    ├── Gradient Boosting
                                    ├── SHAP importance
                                    └── Spillover scores
                                             │
                                             ▼
                                    phylogenetics.py
                                    ├── Distance matrix
                                    └── NJ tree
```

### Snakemake DAG : [Pipeline DAG](results/pipeline_dag.png)

## Key Results
1. Resistance Rates by Organism (Interface-colored)
| Organism | Interface | Resistance Rate |
|---|---|---|
| *A. baumannii* | Environment | **61.3%** ⚠️ |
| *P. aeruginosa* | Environment | **46.8%** ⚠️ |
| *S. aureus* | Animal | 32.5% |
| *E. coli* | Human | 23.1% |
| *S. enterica* | Human | 19.0% |

**Finding: environmental bacteria carry the highest resistance burden — suggesting environment as the primary source of AMR gene flow toward humans.**

2. AMR Co-Resistance Network
- **Nodes:** 20+ antibiotics
- **Edges:** Co-resistance pairs (genomes resistant to both)
- **Top AMR hubs:** Ciprofloxacin (0.72), Ceftriaxone (0.68), Gentamicin (0.65)
- **Implication:** Fluoroquinolones and 3rd-gen cephalosporins are central to MDR spread

3. AMR Gene Flow Network
- Blue nodes = antibiotics with shared resistance across ALL three interfaces
- Identifies cross-interface resistance bridges
- 40+ antibiotics show resistance spanning Human + Animal + Environment

4. Resistance Landscape Heatmap : Cross-interface comparison of 103 antibiotics across Human, Animal, and Environment interfaces.

5. Evolutionary Distance Matrix
| Closest Pair | Distance | Implication |
| S.aureus ↔ S.enterica | **3.43** | Animal→Human transfer risk |
| P.aeruginosa ↔ A.baumannii | 4.68 | Environment internal spread |
| E.coli ↔ S.aureus | 4.40 | Human←Animal spillover |

6. ML Model Performance

| Model | AUC | Notes |
| Random Forest | **0.808** | Best overall |
| Gradient Boosting | 0.791 | Faster inference |

**SHAP Feature Importance:**
Top drivers of AMR resistance prediction:
1. `ab_encoded` — Which antibiotic (0.44) — most important
2. `interface_resist_rate` — Resistance rate at that interface (0.17)
3. `ab_centrality` — Network centrality of antibiotic (0.15)
4. `org_encoded` — Organism identity (0.13)
5. `interface_encoded` — Human/Animal/Environment (0.11)

7. AMR Spillover Risk Scores : Top antibiotics by spillover risk (network centrality × cross-interface resistance):

| Rank | Antibiotic | Risk Score | Class |
| 1 | Ciprofloxacin | 0.46 | Fluoroquinolone |
| 2 | Ceftriaxone | 0.45 | Cephalosporin |
| 3 | Levofloxacin | 0.44 | Fluoroquinolone |
| 4 | Gentamicin | 0.40 | Aminoglycoside |
| 5 | Ceftazidime | 0.36 | Cephalosporin |

**Ciprofloxacin** has the highest spillover risk — a WHO critically important antimicrobial used in both human medicine and veterinary practice.

8. Phylogenetic Tree
AMR-profile based phylogeny colored by interface:
- 🔴 **Red** = Human interface (E.coli, S.enterica)
- 🟠 **Orange** = Animal interface (S.aureus)
- 🟢 **Green** = Environment interface (A.baumannii, P.aeruginosa)

---

## Installation

### Prerequisites
- Python 3.12+
- Git

### Clone Repository
```bash
git clone https://github.com/HSPrithvi23/AMR-EvoNet.git
cd AMR-EvoNet
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
networkx
biopython
streamlit
plotly
snakemake
pyyaml
shap
```

---

## Usage

### Option 1: Run Full Snakemake Pipeline

```bash
# Dry run first (see what will execute)
snakemake -n

# Run full pipeline
snakemake --cores 4

# Generate DAG visualization
snakemake --dag | dot -Tpng > results/pipeline_dag.png
```

### Option 2: Run Individual Scripts

```bash
# Step 1: Data integration
python data_integration.py

# Step 2: Per-organism analysis
python per_organism_analysis.py

# Step 3: Network analysis
python network_analysis.py

# Step 4: ML risk scoring
python ml_risk_scoring.py

# Step 5: Phylogenetics
python phylogenetics.py
```

### Option 3: Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

Then open: http://localhost:8501

---

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Data file paths
data_files:
  E.coli: "E.coli_BVBRC_genome_amr.csv"
  S.enterica: "S.enterics_BVBRC_genome_amr.csv"
  S.aureus: "Staphylococcus aureus BVBRC_genome_amr.csv"
  A.baumannii: "Acinetobacter baumannii BVBRC_genome_amr.csv"
  P.aeruginosa: "Pseudomonas aeruginosa BVBRC_genome_amr.csv"

# Interface assignment
interfaces:
  E.coli: "Human"
  S.enterica: "Human"
  S.aureus: "Animal"
  A.baumannii: "Environment"
  P.aeruginosa: "Environment"

# Analysis parameters
parameters:
  min_co_resistance: 50       # Minimum co-resistance count for network edges
  resistance_threshold: 0.3   # Minimum resistance rate to include in analysis
  n_estimators: 100           # Random Forest trees
  test_size: 0.2              # Train/test split ratio
  random_state: 42            # Reproducibility seed
  top_antibiotics: 20         # Top N antibiotics to display per organism
```

---

## Project Structure

```
AMR-EvoNet/
│
├── Snakefile                    # Snakemake workflow definition
├── config.yaml                  # Pipeline configuration
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes raw data CSVs
│
├── data_integration.py          # Step 1: Load & integrate all datasets
├── per_organism_analysis.py     # Step 2: Per-species AMR profiling
├── network_analysis.py          # Step 3: Co-resistance & gene flow networks
├── ml_risk_scoring.py           # Step 4: ML models + spillover risk
├── phylogenetics.py             # Step 5: Evolutionary analysis
├── streamlit_app.py             # Interactive web dashboard
│
└── results/
    ├── resistance_rate_by_organism.png
    ├── organism_amr_profiles.png
    ├── interface_antibiotic_heatmap.png
    ├── interface_resistance_profile.png
    ├── coresistance_network.png
    ├── interface_geneflow_network.png
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── shap_importance.png
    ├── spillover_risk_scores.png
    ├── phylogenetic_tree.png
    ├── evolutionary_distance_matrix.png
    ├── pipeline_dag.png
    ├── E_coli_resistance_plot.png
    ├── S_enterica_resistance_plot.png
    ├── S_aureus_resistance_plot.png
    ├── A_baumannii_resistance_plot.png
    └── P_aeruginosa_resistance_plot.png
```

---

## Key Scientific Findings

1. **Environmental bacteria are AMR reservoirs** - A.baumannii (61.3%) and P.aeruginosa (46.8%) show the highest resistance rates, suggesting environment as the origin of AMR gene flow toward human pathogens.

2. **Animal-to-Human AMR transfer evidence** - S.aureus (Animal) and S.enterica (Human) share the shortest evolutionary distance (3.43) based on resistance profile similarity, indicating active zoonotic AMR spillover.

3. **Ciprofloxacin is the highest spillover risk antibiotic** - Combining highest network centrality with cross-interface resistance, fluoroquinolones require immediate One Health surveillance priority.

4. **40+ antibiotics show cross-interface resistance** - Indicating widespread horizontal gene transfer of AMR determinants across Human-Animal-Environment boundaries.

5. **Network centrality predicts resistance** — `ab_centrality` is the 3rd most important ML feature (SHAP = 0.15), proving network position of an antibiotic influences resistance likelihood.

---

## One Health & Policy Relevance

This pipeline directly supports:

- **WHO Global Action Plan on AMR** - provides integrated surveillance across HAE interfaces
- **NLM AMR Project (ICAR-NIVEDI)** - epidemiological surveillance in Karnataka & Tamil Nadu livestock
- **NITI Aayog AMR Framework** - computational tools for national AMR surveillance
- **FAO-OIE-WHO Tripartite AMR** - One Health surveillance methodology

---

## Author
**HS Prithvi**
MSc Bioinformatics | JSS Academy of Higher Education and Research, Mysuru
- GitHub: [@HSPrithvi23](https://github.com/HSPrithvi23)

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**⭐ If this project helped you, please give it a star!**

*Built with Python 🐍 | Snakemake 🐍 | Streamlit 🎈 | One Health ❤️*

</div>
