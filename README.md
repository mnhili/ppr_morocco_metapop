# PPR Morocco Metapopulation Model

This repository contains data-processing workflows, spatial analyses, and metapopulation modelling tools for studying peste des petits ruminants (PPR) in Morocco.

The project combines outbreak records, administrative boundaries, sheep and goat population estimates, contact-network data, and simulation-based inference to investigate the spatial and temporal dynamics of PPR.

## Workflow

The numbered directories represent the main analysis workflow:

1.  **WAHIS outbreak geocoding**
2.  **Livestock-population extraction from GLW3 data**
3.  **Outbreak and commune preprocessing**
4.  **Spatial and temporal cluster analysis**
5.  **Case-fatality-rate estimation**
6.  **Metapopulation modelling and simulation-based inference**

## Repository layout

``` text
.
├── 1_wahis_geocoding/
├── 2_population_glw3_extraction/
├── 3_preprocessing_inputs/
├── 4_cluster_analysis/
├── 5_cfr_estimation/
├── 6_metapopulation_model/
├── requirements.txt
├── r_environment.md
└── README.md
```

The main analysis directories contain their own scripts, data, notebooks, and
outputs:

``` text
6_metapopulation_model/
├── data/
├── outputs/
├── R0/
├── run_calibration.py
├── src/
├── TARP/
└── visualization/
```

## Main components

### WAHIS outbreak geocoding

`1_wahis_geocoding/` contains the outbreak-record geocoding workflow, including the original geocoding data, Nominatim results, and re-geocoding script.

### GLW3 livestock-population extraction

`2_population_glw3_extraction/` contains sheep and goat population inputs, spatial boundary data, raster files, and the R script used to extract commune-level population estimates.

### Input preprocessing

`3_preprocessing_inputs/` contains scripts for aggregating outbreak data, adding regional information, and preparing commune-status data.

### Cluster analysis

`4_cluster_analysis/` contains the data, R scripts, figures, network outputs, and other results for spatial and temporal cluster analysis.

### Case-fatality-rate estimation

`5_cfr_estimation/` contains the case-fatality-rate notebook and commune- and province-level input data.

### Metapopulation model

`6_metapopulation_model/` contains the main Python model and inference workflow:

- `src/model.py`: Model definition.
- `src/sbi_pipeline.py`: Simulation-based inference pipeline.
- `src/utils.py`: Shared utilities.
- `run_calibration.py`: Calibration entry point.
- `data/`: Model inputs and contact or population matrices.
- `outputs/`: Inference, simulation, and TARP outputs.
- `R0/`: Basic reproduction-number analysis and summaries.
- `TARP/`: TARP diagnostic scripts.
- `visualization/`: Analysis notebooks and figures.

## Python dependencies

The project uses `requirements.txt` as the single Python dependency specification. The tested environment uses Python 3.12.

The project uses PyTorch for simulation-based inference and CPU-only JAX with
BlackJAX and PyMC for CFR sampling. PyTorch may use a CUDA-enabled
installation when available. Package versions are maintained in
`requirements.txt`.

### Install the dependencies

From the repository root:

``` bash
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r requirements.txt
```

The requirements file uses CPU-only JAX. Do not install the `jax[cuda12]`
extra for this project.

### Verify the installation

```bash
python3.12 -c "import torch, jax, blackjax, pymc; print('Python dependencies OK'); print('JAX backend:', jax.default_backend())"
```

The expected JAX backend is `cpu`.

## R dependencies

The repository contains R preprocessing and cluster-analysis scripts. R dependencies are documented separately in `r_environment.md`.

## Running the metapopulation model

Precomputed inference, simulation, and TARP results are already stored under
`6_metapopulation_model/outputs/`. Running the calibration is only necessary
when reproducing the inference from the input data.

From the repository root:

```bash
cd 6_metapopulation_model
python run_calibration.py --workers 12 --rounds 2000 1000 500
```

This command runs the full calibration and may take a long time. Check the
configuration and input files before starting it. Results are written to
`6_metapopulation_model/outputs/`.
