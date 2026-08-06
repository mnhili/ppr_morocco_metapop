# Load required library
library(dplyr)
library(readxl)
library(writexl)

setwd("D:/PhD_Central/Dual-summary-stat")

# Read the Excel file
data <- read_excel("epi_2008_geocoded.xlsx")

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
write_xlsx(aggregated_data, "ppr_MA_2008_outbreaks_v2.xlsx")

aggregated_commune_day <- data %>%
  group_by(`Start Date`, REGION , PROVINCE, COMMUNE) %>%
  summarise(
    Susceptible = sum(Susceptible, na.rm = TRUE),
    Cases = sum(Cases, na.rm = TRUE),
    Deaths = sum(Deaths, na.rm = TRUE),
    .groups = "drop"
  )

# Save to a new Excel file
write_xlsx(aggregated_commune_day, "ppr_MA_2008_commune_agg_v2.xlsx")

# Count outbreaks per Commune (total outbreaks regardless of date)
outbreaks_per_commune <- data %>%
  group_by(COMMUNE) %>%
  summarise(total_outbreaks = n(), .groups = "drop")

# Count outbreaks per Commune per day
outbreaks_per_commune_day <- data %>%
  group_by(COMMUNE, `Start Date`) %>%
  summarise(outbreaks_same_day = n(), .groups = "drop")


