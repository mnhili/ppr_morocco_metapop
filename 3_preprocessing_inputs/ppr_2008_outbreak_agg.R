# Load required library
library(dplyr)
library(readxl)
library(writexl)

script_path <- sub(
  "^--file=", "",
  commandArgs(trailingOnly = FALSE)[
    grepl("^--file=", commandArgs(trailingOnly = FALSE))
  ][1]
)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
input_file <- file.path(
  repo_root, "4_cluster_analysis", "1_Data",
  "epi_2008_geocoded_Nominatim.xlsx"
)
stopifnot(file.exists(input_file))

# Read the Excel file
data <- read_excel(input_file)

# Aggregate by date, locality, and Commune
aggregated_data <- data %>%
  group_by(`Start Date`, REGION, PROVINCE, COMMUNE, Location, Latitude, Longitude) %>%
  summarise(
    Susceptible = sum(Susceptible, na.rm = TRUE),
    Cases = sum(Cases, na.rm = TRUE),
    Deaths = sum(Deaths, na.rm = TRUE),
    .groups = "drop"
  )

# Save to a new Excel file
write_xlsx(
  aggregated_data,
  file.path(repo_root, "6_metapopulation_model", "data",
            "ppr_MA_2008_outbreaks_v2.xlsx")
)

aggregated_commune_day <- data %>%
  group_by(`Start Date`, REGION , PROVINCE, COMMUNE) %>%
  summarise(
    Susceptible = sum(Susceptible, na.rm = TRUE),
    Cases = sum(Cases, na.rm = TRUE),
    Deaths = sum(Deaths, na.rm = TRUE),
    .groups = "drop"
  )

# Save to a new Excel file
write_xlsx(
  aggregated_commune_day,
  file.path(repo_root, "6_metapopulation_model", "data",
            "ppr_MA_2008_commune_agg_v2.xlsx")
)

# Count outbreaks per Commune (total outbreaks regardless of date)
outbreaks_per_commune <- data %>%
  group_by(COMMUNE) %>%
  summarise(total_outbreaks = n(), .groups = "drop")

# Count outbreaks per Commune per day
outbreaks_per_commune_day <- data %>%
  group_by(COMMUNE, `Start Date`) %>%
  summarise(outbreaks_same_day = n(), .groups = "drop")

