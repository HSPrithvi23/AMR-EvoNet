# AMR-EvoNet Snakemake Pipeline
# Config + Wildcards implementation
# Run: snakemake --cores 4

configfile: "config.yaml"

# ── Pull values from config ───────────────────────────────────
RESULTS  = config["results_dir"]
SCRIPTS  = config["scripts_dir"]
DATA     = config["data_dir"]
PYTHON   = config["python"]
ORGANISMS = config["organisms"]
PARAMS   = config["parameters"]

# ── Final targets ─────────────────────────────────────────────
rule all:
    input:
        # Global pipeline outputs
        f"{RESULTS}/combined_amr_data.csv",
        f"{RESULTS}/network_metrics.csv",
        f"{RESULTS}/spillover_risk_scores.csv",
        f"{RESULTS}/evolutionary_distances.csv",
        f"{RESULTS}/coresistance_network.png",
        f"{RESULTS}/roc_curves.png",
        f"{RESULTS}/phylogenetic_tree.png",
        # Per-organism outputs (wildcards expand to all 5)
        expand(
            f"{RESULTS}/{{organism}}_resistance_profile.csv",
            organism=ORGANISMS
        ),
        expand(
            f"{RESULTS}/{{organism}}_resistance_plot.png",
            organism=ORGANISMS
        )

# ── Rule 1: Data Integration ──────────────────────────────────
rule data_integration:
    input:
        expand(
            f"{DATA}/{{file}}",
            file=list(config["input_files"].values())
        )
    output:
        combined = f"{RESULTS}/combined_amr_data.csv",
        plot1    = f"{RESULTS}/interface_resistance_profile.png",
        plot2    = f"{RESULTS}/resistance_rate_by_organism.png"
    log:
        f"{RESULTS}/logs/01_data_integration.log"
    benchmark:
        f"{RESULTS}/benchmarks/01_data_integration.txt"
    shell:
        '"{PYTHON}" "{SCRIPTS}/01_data_integration.py" '
        '> "{log}" 2>&1'

# ── Rule 2: Per-Organism Analysis (WILDCARD RULE) ─────────────
rule per_organism_analysis:
    input:
        f"{RESULTS}/combined_amr_data.csv"
    output:
        csv = f"{RESULTS}/{{organism}}_resistance_profile.csv",
        png = f"{RESULTS}/{{organism}}_resistance_plot.png"
    params:
        interface = lambda wildcards: config["interfaces"][wildcards.organism]
    log:
        f"{RESULTS}/logs/{{organism}}_analysis.log"
    benchmark:
        f"{RESULTS}/benchmarks/{{organism}}_analysis.txt"
    shell:
        '"{PYTHON}" "{SCRIPTS}/per_organism_analysis.py" '
        '"{wildcards.organism}" '
        '"{params.interface}" '
        '"{input}" '
        '"{output.csv}" '
        '"{output.png}" '
        '> "{log}" 2>&1'

# ── Rule 3: Network Analysis ──────────────────────────────────
rule network_analysis:
    input:
        f"{RESULTS}/combined_amr_data.csv"
    output:
        metrics  = f"{RESULTS}/network_metrics.csv",
        conet    = f"{RESULTS}/coresistance_network.png",
        flownet  = f"{RESULTS}/interface_geneflow_network.png"
    log:
        f"{RESULTS}/logs/02_network_analysis.log"
    benchmark:
        f"{RESULTS}/benchmarks/02_network_analysis.txt"
    shell:
        '"{PYTHON}" "{SCRIPTS}/02_network_analysis.py" '
        '> "{log}" 2>&1'

# ── Rule 4: ML Risk Scoring ───────────────────────────────────
rule ml_risk_scoring:
    input:
        data    = f"{RESULTS}/combined_amr_data.csv",
        network = f"{RESULTS}/network_metrics.csv"
    output:
        scores  = f"{RESULTS}/spillover_risk_scores.csv",
        roc     = f"{RESULTS}/roc_curves.png",
        cm      = f"{RESULTS}/confusion_matrix.png",
        shap    = f"{RESULTS}/shap_importance.png"
    log:
        f"{RESULTS}/logs/03_ml_risk_scoring.log"
    benchmark:
        f"{RESULTS}/benchmarks/03_ml_risk_scoring.txt"
    shell:
        '"{PYTHON}" "{SCRIPTS}/03_ml_risk_scoring.py" '
        '> "{log}" 2>&1'

# ── Rule 5: Phylogenetics ─────────────────────────────────────
rule phylogenetics:
    input:
        data   = f"{RESULTS}/combined_amr_data.csv",
        scores = f"{RESULTS}/spillover_risk_scores.csv"
    output:
        dist   = f"{RESULTS}/evolutionary_distances.csv",
        tree   = f"{RESULTS}/phylogenetic_tree.png",
        matrix = f"{RESULTS}/evolutionary_distance_matrix.png",
        heatmap = f"{RESULTS}/organism_amr_profiles.png"
    log:
        f"{RESULTS}/logs/04_phylogenetics.log"
    benchmark:
        f"{RESULTS}/benchmarks/04_phylogenetics.txt"
    shell:
        '"{PYTHON}" "{SCRIPTS}/04_phylogenetics.py" '
        '> "{log}" 2>&1'