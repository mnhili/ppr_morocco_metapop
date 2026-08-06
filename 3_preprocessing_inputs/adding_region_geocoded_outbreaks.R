library(readxl)
library(dplyr)
library(writexl)

setwd("D:/PhD_Central/Dual-summary-stat")

f1 <- read_excel("ppr_MA_2008_outbreaks.xlsx")
f2 <- read_excel("ppr_MA_2008_region_commune_agg.xlsx")

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
write_xlsx(f1_out, "ppr_MA_2008_outbreaks_with_REGION_PROVINCE_COMMUNE.xlsx")
