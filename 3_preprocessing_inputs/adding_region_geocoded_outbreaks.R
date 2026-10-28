library(readxl)
library(dplyr)
library(writexl)

script_path <- sub(
  "^--file=", "",
  commandArgs(trailingOnly = FALSE)[
    grepl("^--file=", commandArgs(trailingOnly = FALSE))
  ][1]
)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
outbreaks_file <- file.path(
  repo_root, "3_preprocessing_inputs", "data",
  "ppr_MA_2008_outbreaks.xlsx"
)
region_file <- file.path(
  repo_root, "6_metapopulation_model", "data",
  "ppr_MA_2008_region_commune_agg.xlsx"
)
stopifnot(file.exists(outbreaks_file), file.exists(region_file))

f1 <- read_excel(outbreaks_file)
f2 <- read_excel(region_file)

# commune–region lookup
f2_map <- f2 %>%
  select(COMMUNE, REGION) %>%
  distinct()

# join, reorder, rename
f1_out <- f1 %>%
  left_join(f2_map, by = c("Commune Rurale" = "COMMUNE")) %>%
  relocate(REGION, .before = Province) %>%
  rename(
    PROVINCE = Province,
    COMMUNE  = `Commune Rurale`
  )

# export
write_xlsx(
  f1_out,
  file.path(repo_root, "6_metapopulation_model", "data",
            "ppr_MA_2008_outbreaks_with_REGION_PROVINCE_COMMUNE.xlsx")
)
