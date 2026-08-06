# R Environment

This project uses a separate R environment from the Python environment.

## R version

- R 4.3.3

## Project packages used by the R code

- readxl
- dplyr
- writexl
- stringi
- terra
- sf
- exactextractr
- openxlsx
- spacetime
- sp
- xts
- spdep
- reshape2
- nngeo
- mapview
- aqp
- NbClust
- factoextra
- geosphere
- purrr
- tidyr
- outbreaker2
- o2ools
- gridExtra
- dbscan
- tmap
- tmap.glyphs
- cowplot
- GGally
- patchwork
- viridis
- forcats
- igraph
- knitr

## Notes

- The visible R installation in this workspace did not have these project packages installed, so this file is a clean requirements note rather than a lockfile.
- If you want a fully reproducible R lockfile later, the next step is to create an `renv` project in the workspace and record the package versions from the intended R environment.