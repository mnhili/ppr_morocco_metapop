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
│   ├── README.md
│   ├── regeocode_2026.py
│   └── outbreak geocoding data
├── 2_population_glw3_extraction/
│   ├── README.md
│   ├── GIS_data/
│   ├── dataverse_files_Gt_2010/
│   ├── dataverse_files_Gt_2015/
│   ├── dataverse_files_Sh_2010/
│   ├── dataverse_files_Sh_2015/
│   └── sheep_all_communes.R
├── 3_preprocessing_inputs/
│   ├── adding_region_geocoded_outbreaks.R
│   ├── ppr_2008_outbreak_agg.R
│   └── prep_commune_status.R
├── 4_cluster_analysis/
│   ├── 1_Data/
│   ├── 2_Scripts/
│   └── 3_Outputs/
├── 5_cfr_estimation/
│   ├── Case_fatality_rate.ipynb
│   ├── data_commune.csv
│   └── data_province.csv
├── 6_metapopulation_model/
│   ├── data/
│   ├── diagnostics/
│   ├── outputs/
│   ├── run_calibration.py
│   └── src/
├── requirements.txt
├── r_environment.md
└── README.md
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
- `diagnostics/`: Diagnostic scripts and TARP results.
- `outputs/`: Inference, simulation, and TARP outputs.

## Python dependencies

The project uses `requirements.txt` as the single Python dependency specification. Development and testing were performed in Ubuntu running under WSL2 on Windows with Python 3.12.

The tested setup uses GPU-enabled PyTorch, CPU-only JAX, BlackJAX, and PyMC. Their package versions are maintained in `requirements.txt`.

### Install the dependencies

From the repository root:

``` bash
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -r requirements.txt
```

The requirements file should use CPU-only JAX. Do not use the `jax[cuda12]` extra for this project. PyTorch may use the available NVIDIA GPU, while JAX remains on the CPU.

### Verify the installation

``` bash
python3.12 - <<'PY'
import blackjax
import jax
import pymc
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("PyTorch CUDA available:", torch.cuda.is_available())
print("JAX:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
print("BlackJAX:", blackjax.__version__)
print("PyMC:", pymc.__version__)

if torch.cuda.is_available():
    print("PyTorch GPU:", torch.cuda.get_device_name(0))
PY
```

For WSL2 GPU support, the NVIDIA driver is installed on the Windows host. A separate Linux NVIDIA display driver should not be installed inside WSL2.

## R dependencies

The repository contains R preprocessing and cluster-analysis scripts. R dependencies are documented separately in `r_environment.md`.

## Running the metapopulation model

From the repository root:

``` bash
cd 6_metapopulation_model
python run_calibration.py
```

Check the configuration and input files before starting a long calibration. Results are written to `6_metapopulation_model/outputs/` according to the selected workflow.
