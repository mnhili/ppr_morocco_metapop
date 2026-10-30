library(terra)
library(sf)
library(exactextractr)
library(openxlsx)
library(dplyr)

script_path <- sub(
  "^--file=", "",
  commandArgs(trailingOnly = FALSE)[
    grepl("^--file=", commandArgs(trailingOnly = FALSE))
  ][1]
)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
population_root <- file.path(repo_root, "2_population_glw3_extraction")
output_file <- file.path(
  population_root, "sheep_goat_populations_communes_GLW_v2.xlsx"
)


# Rasters
sheep_raster_2015 <- rast(file.path(
  population_root, "dataverse_files_Sh_2015", "5_Sh_2015_Da.tif"
))
sheep_raster_2010 <- rast(file.path(
  population_root, "dataverse_files_Sh_2010", "5_Sh_2010_Da.tif"
))
goat_raster_2015 <- rast(file.path(
  population_root, "dataverse_files_Gt_2015", "5_Gt_2015_Da.tif"
))
goat_raster_2010 <- rast(file.path(
  population_root, "dataverse_files_Gt_2010", "5_Gt_2010_Da.tif"
))

# Shapefile
morocco_shp <- st_read(file.path(
  population_root, "GIS_data", "New_SHP_Morocco.shp"
))

# CRS harmonisation
morocco_shp <- st_transform(morocco_shp, crs = st_crs(sheep_raster_2015))

# Extract populations
morocco_shp$sheep_2015 <- exact_extract(sheep_raster_2015, morocco_shp, "sum")
morocco_shp$sheep_2010 <- exact_extract(sheep_raster_2010, morocco_shp, "sum")
morocco_shp$goat_2015  <- exact_extract(goat_raster_2015,  morocco_shp, "sum")
morocco_shp$goat_2010  <- exact_extract(goat_raster_2010,  morocco_shp, "sum")

names(morocco_shp)


# Final table
results <- morocco_shp |>
  st_drop_geometry() |>
  select(
    REGION,
    COMMUNE,
    sheep_2010,
    sheep_2015,
    goat_2010,
    goat_2015
  )


print(head(results))

results <- results %>%
  arrange(
    stringi::stri_trans_general(REGION, "Latin-ASCII"),
    stringi::stri_trans_general(COMMUNE, "Latin-ASCII")
  )


write.xlsx(
  results,
  file = output_file,
  rowNames = FALSE
)

