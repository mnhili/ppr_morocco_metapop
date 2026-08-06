# CFR Estimation

This folder contains the material used to estimate the case-fatality-rate parameter for PPR using NUTS MCMC.

## Contents

- `Case_fatality_rate.ipynb` estimation notebook.
- `data_commune.csv` commune-level input data.
- `data_province.csv` province-level input data.

## How it connects

- The estimated mean CFR is used as a fixed input in the metapopulation model.
- The estimation is separate from the main model calibration workflow because the likelihood is tractable here.

