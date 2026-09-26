"""
SBI calibration pipeline.

Parameter vector layout (11 dimensions):
  [0..3]  beta_intra  per group  (Casablanca-Settat, Fes-Meknes, Other, Rabat-Beni)
  [4..7]  beta_local  per group  (same order)
  [8]     beta_inter
  [9]     k   (NegBin dispersion)
  [10]    rho (importation reporting rate)

Summary statistic: first ISO week where Z exceeds 90 % of observed first-week cases,
one value per infected commune.
"""

import time
import numpy as np
import torch
from sbi.inference import SNPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform, RestrictedPrior, get_density_thresholder

from src.utils import GROUP_NAMES, region_to_group
from src.model import run_simulation, seird_ode

N_GROUPS = len(GROUP_NAMES)  # 4
N_PARAMS = 2 * N_GROUPS + 3  # 11


# ---------------------------------------------------------------------------
# Wrapper: theta -> SS vector
# ---------------------------------------------------------------------------

def make_simulator(study: dict, model_params: dict):
    """
    Return a function  simulator(theta_batch) -> x_batch
    that is compatible with sbi.simulate_for_sbi.

    study      : output of prepare_study_data()
    model_params : dict with sigma, gamma, mu, total_days, week
    """
    population = study["population"]
    n_patches = study["n_patches"]
    patch_regions = study["patch_regions"]
    patch_names = study["patch_names"]
    initial_state = study["initial_state"]
    observed_90pct = study["observed_90pct"]
    infected_sorted = sorted(study["infected_patches"])
    contact_matrix = study.get("contact_matrix")  # None for full connectivity

    sigma = model_params["sigma"]
    gamma = model_params["gamma"]
    mu = model_params["mu"]
    total_days = model_params["total_days"]
    week = model_params["week"]
    total_weeks = total_days // week
    MISS_WEEK = float(24 + total_weeks + 1)  # ≈ 50 for 180 days

    def _unpack(theta_1d):
        params = {}
        for j, g in enumerate(GROUP_NAMES):
            params[f"beta_intra_{g}"] = theta_1d[j].item()
        for j, g in enumerate(GROUP_NAMES):
            params[f"beta_local_{g}"] = theta_1d[N_GROUPS + j].item()
        params["beta_inter"] = theta_1d[2 * N_GROUPS].item()
        params["k"] = theta_1d[2 * N_GROUPS + 1].item()
        params["rho"] = theta_1d[2 * N_GROUPS + 2].item()
        return params

    def _simulate_one(theta_1d):
        p = _unpack(theta_1d)

        beta_local = {g: p[f"beta_local_{g}"] for g in GROUP_NAMES}
        beta_intra = {g: p[f"beta_intra_{g}"] for g in GROUP_NAMES}

        weekly_df = run_simulation(
            beta_local=beta_local,
            beta_intra=beta_intra,
            beta_inter=p["beta_inter"],
            population=population,
            n_patches=n_patches,
            total_days=total_days,
            week=week,
            patch_regions=patch_regions,
            initial_state=initial_state,
            patch_names=patch_names,
            sigma=sigma,
            gamma=gamma,
            mu=mu,
            observed_90pct=observed_90pct,
            k=p["k"],
            rho=p["rho"],
            contact_matrix=contact_matrix,
        )

        # SS first ISO week with Z > threshold per infected commune
        ss = []
        for commune, region in infected_sorted:
            mask = (weekly_df["COMMUNE"] == commune) & (weekly_df["REGION"] == region)
            patch_data = weekly_df[mask].sort_values("week_end_day")
            if patch_data.empty:
                ss.append(MISS_WEEK)
                continue
            threshold = observed_90pct.get((commune, region), float("inf"))
            valid = patch_data[patch_data["Z"] > threshold]
            ss.append(float(valid["iso_week"].min()) if not valid.empty else MISS_WEEK)

        return ss

    def simulator(params_batch):
        if isinstance(params_batch, np.ndarray):
            params_batch = torch.from_numpy(params_batch).float()
        if params_batch.ndim == 1:
            params_batch = params_batch.unsqueeze(0)

        results = []
        for i in range(params_batch.shape[0]):
            try:
                ss = _simulate_one(params_batch[i])
            except Exception:
                ss = [MISS_WEEK] * len(infected_sorted)
            results.append(ss)

        return torch.as_tensor(np.array(results), dtype=torch.float32)

    return simulator, infected_sorted, MISS_WEEK


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def build_prior():
    lo = torch.tensor(
        [0.01] * N_GROUPS       # beta_intra
        + [0.15] * N_GROUPS     # beta_local
        + [0.001]               # beta_inter
        + [0.1]                 # k
        + [0.05]                # rho
    )
    hi = torch.tensor(
        [0.41] * N_GROUPS
        + [0.80] * N_GROUPS
        + [0.101]
        + [5.0]
        + [0.50]
    )
    return BoxUniform(low=lo, high=hi)


def param_names() -> list[str]:
    names = [f"beta_intra_{g}" for g in GROUP_NAMES]
    names += [f"beta_local_{g}" for g in GROUP_NAMES]
    names += ["beta_inter", "k", "rho"]
    return names


def unpack_theta(theta_1d) -> dict:
    """Convert a flat parameter vector into a named dict."""
    p = {}
    for j, g in enumerate(GROUP_NAMES):
        p[f"beta_intra_{g}"] = float(theta_1d[j])
    for j, g in enumerate(GROUP_NAMES):
        p[f"beta_local_{g}"] = float(theta_1d[N_GROUPS + j])
    p["beta_inter"] = float(theta_1d[2 * N_GROUPS])
    p["k"] = float(theta_1d[2 * N_GROUPS + 1])
    p["rho"] = float(theta_1d[2 * N_GROUPS + 2])
    return p


# ---------------------------------------------------------------------------
  """
    Truncated Sequential NPE (TSNPE).

    It Uses ``RestrictedPrior`` + ``get_density_thresholder`` from sbi
    (canonical implementation).  Intermediate posteriors are built with
    MCMC so that ``get_density_thresholder`` can draw samples reliably.
    All rounds use ``force_first_round_loss=True``.
    """
def run_snpe(
    simulator_fn,
    prior,
    x_obs: torch.Tensor,
    rounds: list[int] | None = None,
    num_workers: int = 4,
    tsnpe_quantile: float = 1e-4,
    **_kwargs,
) -> tuple:
  
    if rounds is None:
        rounds = [2000, 1000, 500]

    inference = SNPE(prior=prior, density_estimator=posterior_nn(model="nsf"))
    proposal = prior
    theta_all, x_all = [], []
    density_est = None

    t0 = time.time()

    for r, n_sim in enumerate(rounds, 1):
        tag = "prior" if r == 1 else "restricted prior"
        print(f"[TSNPE Round {r}/{len(rounds)}] "
              f"Simulating {n_sim} from {tag} ...")

        theta_r, x_r = simulate_for_sbi(
            simulator_fn, proposal=proposal,
            num_simulations=n_sim, num_workers=num_workers,
        )
        theta_all.append(theta_r)
        x_all.append(x_r)

        density_est = inference.append_simulations(
            theta_r, x_r
        ).train(force_first_round_loss=True)

        n_total = sum(t.shape[0] for t in theta_all)
        print(f"  -> Round {r} training done ({n_total} total sims).")

        # Restrict prior for next round
        if r < len(rounds):
            posterior_tmp = inference.build_posterior(
                density_est,
                sample_with="mcmc",
                mcmc_method="slice_np_vectorized",
            ).set_default_x(x_obs)

            accept_reject_fn = get_density_thresholder(
                posterior_tmp, quantile=tsnpe_quantile,
                num_samples_to_estimate_support=5_000,
            )
            proposal = RestrictedPrior(
                prior, accept_reject_fn, sample_with="rejection",
            )
            print(f"  -> Prior restricted for round {r + 1}.")

    posterior = inference.build_posterior(
        density_est,
        sample_with="mcmc",
        mcmc_method="slice_np_vectorized",
    )

    elapsed = time.time() - t0
    theta_all = torch.cat(theta_all)
    x_all_cat = torch.cat(x_all)

    print(f"\nTSNPE finished in {elapsed / 60:.1f} min "
            f"({sum(rounds)} total simulations across {len(rounds)} rounds, "
            f"sampler=mcmc, flow=nsf).")

    return posterior, (theta_all, x_all_cat), elapsed
