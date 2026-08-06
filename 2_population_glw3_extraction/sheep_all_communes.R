library(terra)
library(sf)
library(exactextractr)
library(openxlsx)
library(dplyr)

setwd("C:/Users/manhi/OneDrive/Documents/Population_V2")


# Rasters
sheep_raster_2015 <- rast("dataverse_files_Sh_2015/5_Sh_2015_Da.tif")
sheep_raster_2010 <- rast("dataverse_files_Sh_2010/5_Sh_2010_Da.tif")
goat_raster_2015  <- rast("dataverse_files_Gt_2015/5_Gt_2015_Da.tif")
goat_raster_2010  <- rast("dataverse_files_Gt_2010/5_Gt_2010_Da.tif")

# Shapefile
morocco_shp <- st_read("GIS_data/New_SHP_Morocco.shp")

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
  file = "sheep_goat_populations_communes_GLW_v2.xlsx",
  rowNames = FALSE
)


