library(readxl)
library(dplyr)
library(stringi)
library(writexl)

script_path <- sub(
  "^--file=", "",
  commandArgs(trailingOnly = FALSE)[
    grepl("^--file=", commandArgs(trailingOnly = FALSE))
  ][1]
)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
regions_file <- file.path(
  repo_root, "2_population_glw3_extraction", "GIS_data",
  "regions_provinces_communes.xlsx"
)
outbreaks_file <- file.path(
  repo_root, "4_cluster_analysis", "1_Data",
  "epi_2008_geocoded_Nominatim.xlsx"
)
stopifnot(file.exists(regions_file), file.exists(outbreaks_file))

# Read files
file1 <- read_excel(regions_file)
file2 <- read_excel(outbreaks_file)

# Normalise names for matching (no accents, case-insensitive)
file1 <- file1 %>%
  mutate(COMMUNE_key = stri_trans_general(COMMUNE, "Latin-ASCII") |> tolower())

file2 <- file2 %>%
  mutate(Commune_key = stri_trans_general(`Commune Rurale`, "Latin-ASCII") |> tolower())

# Add STATUS column
file1 <- file1 %>%
  mutate(
    STATUS = ifelse(COMMUNE_key %in% file2$Commune_key, 1, 0)
  ) %>%
  select(-COMMUNE_key)

file1 %>%
  filter(STATUS == 1) %>%
  summarise(n_unique = n_distinct(COMMUNE))


# Export
write_xlsx(
  file1,
  file.path(repo_root, "6_metapopulation_model", "data",
            "regions_provinces_communes_status.xlsx")
)
