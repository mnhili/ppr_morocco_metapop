"""
Data loading, population rounding, and region-group mapping.
"""

import unicodedata

import numpy as np
import pandas as pd
from pathlib import Path


def _strip_accents(s: str) -> str:
    """to remove diacritical marks:  Fés-Meknès  ->  Fes-Meknes"""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

# ---------------------------------------------------------------------------
# Region grouping
# ---------------------------------------------------------------------------

REGION_GROUPS = {
    "Casablanca-Settat": "Casablanca-Settat",
    "Fés-Meknès": "Fes-Meknes",
    "Rabat-Salé-Kénitra": "Rabat-Beni",
    "Béni Mellal-Khénifra": "Rabat-Beni",
    "Daraâ-Tafilalet": "Other",
    "Oriental": "Other",
    "Marrakech-Safi": "Other",
    "Souss-Massa": "Other",
    "Tanger-Tétouan-Al Hoceima": "Other",
}

GROUP_NAMES = sorted(set(REGION_GROUPS.values()))  # alphabetical


def region_to_group(region: str) -> str:
    return REGION_GROUPS.get(region, "Other")


# ---------------------------------------------------------------------------
# Contact / adjacency matrix
# ---------------------------------------------------------------------------

def load_contact_matrix(
    filepath: Path,
    patch_names: np.ndarray,
    patch_regions: np.ndarray,
) -> np.ndarray:

    ext = filepath.suffix.lower()
    if ext in (".xlsx", ".xls"):
        edges = pd.read_excel(filepath)
    else:
        edges = pd.read_csv(filepath, encoding="utf-8-sig")

    n = len(patch_names)

    lookup: dict[tuple[str, str], int] = {}
    for idx in range(n):
        key = (_strip_accents(patch_names[idx]), _strip_accents(patch_regions[idx]))
        lookup[key] = idx

    contact = np.zeros((n, n), dtype=np.float64)

    matched = 0
    for _, row in edges.iterrows():
        key_from = (
            _strip_accents(str(row["Commune_from"]).strip()),
            _strip_accents(str(row["Region_from"]).strip()),
        )
        key_to = (
            _strip_accents(str(row["Commune_to"]).strip()),
            _strip_accents(str(row["Region_to"]).strip()),
        )
        i = lookup.get(key_from)
        j = lookup.get(key_to)
        if i is not None and j is not None:
            contact[i, j] = 1.0
            contact[j, i] = 1.0  # symmetric
            matched += 1

    n_edges = int((contact > 0).sum()) // 2  # divide by symmetry
    print(f"Contact matrix (binary): {n}×{n}, {n_edges} edges "
          f"({matched} rows matched from {len(edges)} edge-list rows)")
    return contact


# ---------------------------------------------------------------------------
# Population integerisation (stochastic rounding preserving totals)
# ---------------------------------------------------------------------------

def integerize_population(x: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.floor(x).astype(int)
    remainder = x - base
    deficit = int(round(x.sum() - base.sum()))
    if deficit > 0 and remainder.sum() > 0:
        probs = remainder / remainder.sum()
        idx = rng.choice(len(x), size=deficit, replace=False, p=probs)
        base[idx] += 1
    return base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> dict:
    communes_status = pd.read_excel(data_dir / "regions_communes_status.xlsx")
    outbreaks = pd.read_excel(
        data_dir / "ppr_MA_2008_region_commune_agg.xlsx", sheet_name="Sheet1"
    )
    populations = pd.read_excel(
        data_dir / "sheep_goat_populations_communes_GLW_sorted.xlsx"
    )
    return {
        "communes_status": communes_status,
        "outbreaks": outbreaks,
        "populations": populations,
    }


def prepare_study_data(
    data: dict,
    population_col: str = "sr_2010",
    patches_mode: str = "all",
) -> dict:

    f1 = data["communes_status"].drop_duplicates(subset=["REGION", "COMMUNE"])
    outbreaks = data["outbreaks"].copy()
    f3 = data["populations"].drop_duplicates(subset=["REGION", "COMMUNE"])

    # ISO week
    outbreaks["iso_week"] = outbreaks["Start Date"].dt.isocalendar().week.astype(int)

    # Weekly aggregation
    weekly_obs = (
        outbreaks
        .groupby(["iso_week", "REGION", "COMMUNE"], as_index=False)
        .agg(
            cum_susceptible=("Susceptible", "sum"),
            cum_cases=("Cases", "sum"),
            cum_deaths=("Deaths", "sum"),
        )
    )
    weekly_obs["cases_90pct"] = 0.9 * weekly_obs["cum_cases"]
    weekly_obs = weekly_obs.sort_values(["REGION", "COMMUNE", "iso_week"])

    # First outbreak week per patch
    first_outbreaks = (
        weekly_obs
        .groupby(["REGION", "COMMUNE"], as_index=False)
        .first()[["REGION", "COMMUNE", "iso_week", "cum_susceptible",
                  "cum_cases", "cases_90pct", "cum_deaths"]]
    )

    # Infected patches set
    infected_patches = set(zip(weekly_obs["COMMUNE"], weekly_obs["REGION"]))
    regions_with_infection = set(weekly_obs["REGION"])

    # Merge populations
    patch_df = f1.merge(f3, on=["REGION", "COMMUNE"], how="left")

    # Integerise population columns
    pop_cols = ["sheep_2010", "sheep_2015", "goat_2010", "goat_2015",
                "sr_2010", "sr_2015"]
    for col in pop_cols:
        if col in patch_df.columns:
            patch_df[col] = (
                patch_df
                .groupby(["REGION", "COMMUNE"])[col]
                .transform(lambda s: integerize_population(s.values))
            )

    # Keep only regions that had outbreaks
    patch_df = patch_df[patch_df["REGION"].isin(regions_with_infection)].copy()

    # Attach infection status
    patch_df["STATUS"] = patch_df.apply(
        lambda r: 1 if (r["COMMUNE"], r["REGION"]) in infected_patches else 0,
        axis=1,
    )

    if patches_mode == "infected_only":
        patch_df = patch_df[patch_df["STATUS"] == 1].copy()
        patch_df = patch_df.reset_index(drop=True)

    # Population vector
    population = patch_df[population_col].values

    # 90 % cases threshold dict
    observed_90pct = (
        first_outbreaks
        .set_index(["COMMUNE", "REGION"])["cases_90pct"]
        .to_dict()
    )

    # T1 observed: first ISO week per infected patch
    T1_observed = {}
    for commune, region in infected_patches:
        row = first_outbreaks[
            (first_outbreaks["COMMUNE"] == commune)
            & (first_outbreaks["REGION"] == region)
        ]
        if not row.empty:
            T1_observed[(commune, region)] = int(row["iso_week"].values[0])

    return {
        "patch_df": patch_df,
        "weekly_obs": weekly_obs,
        "first_outbreaks": first_outbreaks,
        "observed_90pct": observed_90pct,
        "infected_patches": infected_patches,
        "T1_observed": T1_observed,
        "population": population,
        "patch_names": np.asarray(patch_df["COMMUNE"].values, dtype=str),
        "patch_regions": np.asarray(patch_df["REGION"].values, dtype=str),
        "n_patches": len(patch_df),
        "patches_mode": patches_mode,
    }
