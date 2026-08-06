library(readxl)
library(dplyr)
library(stringi)
library(writexl)

setwd("D:/PhD_Central/Dual-summary-stat")

# Read files
file1 <- read_excel("regions_provinces_communes.xlsx")
file2 <- read_excel("epi_2008_geocoded.xlsx")

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
write_xlsx(file1, "regions_provinces_communes_status.xlsx")
