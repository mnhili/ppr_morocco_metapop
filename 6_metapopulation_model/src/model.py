"""
SEIRD metapopulation model with region-group-specific transmission rates.

Compartments per patch:  S  E  I  R  D  Z
  S : Susceptible
  E : Exposed (latent)
  I : Infectious
  R : Recovered
  D : Dead
  Z : Cumulative incidence (tracking variable)

Three spatial scales of transmission:
  beta_local  – within a patch             (group-specific)
  beta_intra  – between patches, same region (group-specific)
  beta_inter  – between patches, different regions (scalar)
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from datetime import datetime

from src.utils import region_to_group, GROUP_NAMES

REFERENCE_DATE = datetime(2008, 6, 12)


# ---------------------------------------------------------------------------
# Single-patch ODE
# ---------------------------------------------------------------------------

def seird_ode(state, t, beta, sigma, gamma, mu):
    S, E, I, R, D, Z = state
    N = S + E + I + R
    if N <= 0:
        return [0.0] * 6
    force = beta * S * I / N
    dS = -force
    dE = force - sigma * E
    dI = sigma * E - gamma * I
    dR = (1 - mu) * gamma * I
    dD = mu * gamma * I
    dZ = force
    return [dS, dE, dI, dR, dD, dZ]


# ---------------------------------------------------------------------------
# Transmission matrix  (vectorised construction)
# ---------------------------------------------------------------------------

def build_transmission_matrix(
    patch_regions: np.ndarray,
    beta_local: dict,
    beta_intra: dict,
    beta_inter: float,
    contact_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct n x n transmission-rate matrix.

    Diagonal : beta_local[group(i)]
    Off-diag same region : beta_intra[group(i)]
    Off-diag diff region : beta_inter
    """
    n = len(patch_regions)
    groups = np.array([region_to_group(r) for r in patch_regions])

    # Start with inter-region rate everywhere
    T = np.full((n, n), beta_inter, dtype=np.float64)

    # Same-region mask  (vectorised)
    same_region = patch_regions[:, None] == patch_regions[None, :]

    for g in GROUP_NAMES:
        g_mask = groups == g
        # intra: off-diagonal, same region, same group
        intra_mask = same_region & g_mask[:, None] & g_mask[None, :]
        T[intra_mask] = beta_intra[g]
        # diagonal (local)
        diag_idx = np.where(g_mask)[0]
        T[diag_idx, diag_idx] = beta_local[g]

    # Apply contact mask to off-diagonal entries
    if contact_matrix is not None:
        diag_vals = np.diag(T).copy()
        T *= contact_matrix
        np.fill_diagonal(T, diag_vals)  # preserve diagonal (local rates)

    return T


# ---------------------------------------------------------------------------
# Metapopulation simulator
# ---------------------------------------------------------------------------

def run_simulation(
    beta_local: dict,
    beta_intra: dict,
    beta_inter: float,
    population: np.ndarray,
    n_patches: int,
    total_days: int,
    week: int,
    patch_regions: np.ndarray,
    initial_state: np.ndarray,
    patch_names: np.ndarray,
    sigma: float,
    gamma: float,
    mu: float,
    observed_90pct: dict | None = None,
    k: float = 1.0,
    rho: float = 0.1,
    contact_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Metapopulation SEIRD simulation.

    1. Stochastic inter-patch exposure (negative-binomial importation).
    2. Deterministic within-patch SEIRD dynamics (ODE per patch).

    Returns a DataFrame with columns:
        patch, week_end_day, S, E, I, R, D, Z, COMMUNE, REGION, date, iso_week
    """
    T_mat = build_transmission_matrix(
        patch_regions, beta_local, beta_intra, beta_inter,
        contact_matrix=contact_matrix,
    )

    pop_safe = population.copy().astype(float)
    pop_safe[pop_safe <= 0] = 1.0  # avoid division by zero

    state = initial_state.copy()
    already_exposed = np.zeros(n_patches, dtype=bool)
    weekly_results = []
    day = 1

    while day <= total_days:
        # --- Inter-patch force of infection (vectorised) ---
        I_frac = state[:, 2] / pop_safe                   # I_j / N_j
        T_offdiag = T_mat.copy()
        np.fill_diagonal(T_offdiag, 0.0)
        lambda_global = T_offdiag @ I_frac                 # n-vector

        # --- Stochastic importation for unexposed patches ---
        unexposed = (~already_exposed) & (state[:, 1] + state[:, 2] == 0)
        if unexposed.any():
            idx_unexp = np.where(unexposed)[0]
            S_vals = state[idx_unexp, 0]
            raw_mean = rho * S_vals * lambda_global[idx_unexp]

            for pos, i in enumerate(idx_unexp):
                m = raw_mean[pos]
                if m <= 0 or k <= 0:
                    continue
                p = k / (k + m)
                delta = int(min(np.random.negative_binomial(k, p), S_vals[pos]))
                if delta > 0:
                    already_exposed[i] = True
                    state[i, 0] -= delta
                    state[i, 1] += delta

        # --- Within-patch ODE integration ---
        remaining = min(week, total_days - day + 1)
        t_span = np.linspace(0, remaining, remaining + 1)

        new_state = np.empty_like(state)
        for i in range(n_patches):
            patch = state[i]
            patch_key = (patch_names[i], patch_regions[i])

            # Stop updating if cumulative incidence exceeds 90 % of observed
            if observed_90pct and patch_key in observed_90pct:
                if patch[5] > observed_90pct[patch_key]:
                    new_state[i] = patch
                    continue

            sol = odeint(seird_ode, patch, t_span,
                         args=(T_mat[i, i], sigma, gamma, mu))
            new_state[i] = sol[-1]

        # Record weekly snapshot
        for i in range(n_patches):
            weekly_results.append((
                i, day + remaining - 1,
                *new_state[i],
            ))

        state = new_state
        day += remaining

        if state[:, 1:3].sum() == 0:
            break

    cols = ["patch", "week_end_day", "S", "E", "I", "R", "D", "Z"]
    df = pd.DataFrame(weekly_results, columns=cols)

    df["COMMUNE"] = df["patch"].map(lambda idx: patch_names[idx])
    df["REGION"] = df["patch"].map(lambda idx: patch_regions[idx])
    df["date"] = df["week_end_day"].apply(
        lambda d: REFERENCE_DATE + pd.Timedelta(days=int(d))
    )
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)

    return df
