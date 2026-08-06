#!/usr/bin/env python
"""
PPR metapopulation calibration – single-scenario entry point.

Usage
-----
    python run_calibration.py                         # defaults
    python run_calibration.py --workers 8 --rounds 1000 500 500
    python run_calibration.py --samples 5000 --skip-posterior-sims
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from sbi.inference.posteriors.direct_posterior import DirectPosterior

from src.utils import load_data, prepare_study_data, GROUP_NAMES, load_contact_matrix
from src.model import run_simulation
from src.sbi_pipeline import (
    make_simulator,
    build_prior,
    param_names,
    run_snpe,
    unpack_theta,
    N_GROUPS,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "outputs"
INFERENCE_DIR = RESULTS_DIR / "inference"
SIMULATION_DIR = RESULTS_DIR / "simulations"
PATCHES_MODE = "infected_only"
CONNECTIVITY = "contact"

# Fixed defaults — retained scenario, contact, TSNPE + NSF
SAMPLES = 10_000
ANALYSIS_SAMPLES = 10_000
SKIP_POSTERIOR_SIMS = False
SEED_PATCH = "Ain Chkef"
TOTAL_DAYS = 180
N_TRAJECTORIES = 500
CONTACT_FILE = "data/Commune_Contact_matrix.csv"
SEED = 42
STRATEGY = "tsnpe"
FLOW = "nsf"

def parse_args():
    p = argparse.ArgumentParser(description="PPR SBI calibration (single scenario)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rounds", type=int, nargs="+", default=[1000, 500, 500])
    p.add_argument("--sampler", type=str, default="mcmc",
                   help="Sampler name (keeps for metadata; calibration uses TSNPE)")
    return p.parse_args()


def build_initial_state(study: dict, seed_patch: str) -> np.ndarray:
    n = study["n_patches"]
    population = study["population"]
    patch_names = study["patch_names"]

    state = np.zeros((n, 6))
    state[:, 0] = population - 1

    try:
        idx = list(patch_names).index(seed_patch)
        state[idx, 2] = 1
        print(f"Seeded infection in patch {idx}: {seed_patch}")
    except ValueError:
        state[0, 2] = 1
        print(f"'{seed_patch}' not found – fallback to patch 0: {patch_names[0]}")

    return state


def build_direct_posterior(posterior, prior):
    """Rebuild a direct-sampling posterior for diagnostics."""
    density_estimator = posterior.posterior_estimator
    return DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior,
        device=str(next(density_estimator.parameters()).device),
    )


def main():
    args = parse_args()

    # ---- Reproducibility seeds ----------------------------------------------
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"Random seed: {SEED}")

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    SIMULATION_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load & prepare data ------------------------------------------------
    print(f"Loading data  (patches_mode={PATCHES_MODE}) ...")
    raw = load_data(DATA_DIR)
    study = prepare_study_data(raw, patches_mode=PATCHES_MODE)

    params_csv = pd.read_csv(DATA_DIR / "model_parameters.csv")
    model_cfg = dict(zip(params_csv["Parameter"], params_csv["Value"]))

    model_params = {
        "sigma": model_cfg["sigma"],
        "gamma": model_cfg["gamma"],
        "mu": model_cfg["mu"],
        "total_days": TOTAL_DAYS,
        "week": int(model_cfg["week"]),
    }

    study["initial_state"] = build_initial_state(study, SEED_PATCH)

    # ---- Contact matrix (required for the retained contact scenario) ------
    cpath = Path(CONTACT_FILE)
    if not cpath.exists():
        raise SystemExit(f"ERROR: contact file not found: {cpath}")
    study["contact_matrix"] = load_contact_matrix(
        cpath, study["patch_names"], study["patch_regions"],
    )

    print(f"Patches: {study['n_patches']}  |  "
          f"Infected: {len(study['infected_patches'])}  |  "
          f"Regions: {len(set(study['patch_regions']))}")

    # ---- Build SBI components -----------------------------------------------
    simulator_fn, infected_sorted, MISS_WEEK = make_simulator(study, model_params)
    prior = build_prior()

    # Observed SS2 vector  (use same sentinel as the simulator)
    T1_obs = study["T1_observed"]
    x_obs = torch.tensor(
        [T1_obs.get(p, MISS_WEEK) for p in infected_sorted], dtype=torch.float32
    )
    print(f"SS2 dimension: {len(x_obs)}")

    # ---- SNPE ---------------------------------------------------------------
    posterior, (theta_all, x_all), elapsed = run_snpe(
        simulator_fn, prior, x_obs,
        rounds=args.rounds, num_workers=args.workers,
    )

    # ---- Posterior samples ---------------------------------------------------
    print(f"Drawing {SAMPLES} posterior samples ...")
    samples = posterior.sample((SAMPLES,), x=x_obs)
    samples_np = samples.detach().numpy()
    posterior_df = pd.DataFrame(samples_np, columns=param_names())

    posterior_df.to_csv(INFERENCE_DIR / "posterior_samples.csv", index=False,
                        encoding="utf-8-sig")
    print(f"  saved -> {INFERENCE_DIR.relative_to(ROOT)}/posterior_samples.csv")

    with open(INFERENCE_DIR / "posterior.pkl", "wb") as f:
        pickle.dump(posterior, f)
    print(f"  saved -> {INFERENCE_DIR.relative_to(ROOT)}/posterior.pkl")

    try:
        posterior_direct = build_direct_posterior(posterior, prior)
        with open(INFERENCE_DIR / "posterior_direct.pkl", "wb") as f:
            pickle.dump(posterior_direct, f)
        print(f"  saved -> {INFERENCE_DIR.relative_to(ROOT)}/posterior_direct.pkl")
    except Exception as exc:
        print(f"  WARNING: could not save posterior_direct.pkl ({exc})")

    # ---- Posterior predictive simulations ------------------------------------
    if not SKIP_POSTERIOR_SIMS:
        n_analysis = min(ANALYSIS_SAMPLES, len(posterior_df))
        indices = np.random.choice(len(posterior_df), size=n_analysis, replace=False)
        analysis_theta = samples[indices]

        print(f"Running {n_analysis} posterior predictive simulations ...")

        def _run_one(i):
            theta_i = analysis_theta[i]
            out = simulator_fn(theta_i.unsqueeze(0))
            return out.squeeze(0).numpy()

        ss2_preds = list(Parallel(n_jobs=args.workers)(
            delayed(_run_one)(i) for i in tqdm(range(n_analysis), desc="Posterior sims")
        ))

        ss2_cols = [f"{c} | {r}" for c, r in infected_sorted]
        ss2_df = pd.DataFrame(ss2_preds, columns=ss2_cols).round().astype(int)
        ss2_df.to_csv(SIMULATION_DIR / "ss2_posterior_predictions.csv", index=False,
                      encoding="utf-8-sig")
        print(f"  saved -> {SIMULATION_DIR.relative_to(ROOT)}/ss2_posterior_predictions.csv")

        # Posterior stats table
        stats_rows = []
        for j, patch in enumerate(infected_sorted):
            obs_val = T1_obs.get(patch, MISS_WEEK)
            valid = ss2_df.iloc[:, j][ss2_df.iloc[:, j] < MISS_WEEK]
            if len(valid) == 0:
                continue
            pred_mean = valid.mean()
            ci_lo = np.percentile(valid, 2.5)
            ci_hi = np.percentile(valid, 97.5)
            stats_rows.append({
                "COMMUNE": patch[0],
                "REGION": patch[1],
                "observed_T1": obs_val,
                "predicted_mean": pred_mean,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "within_95ci": ci_lo <= obs_val <= ci_hi,
                "abs_error": abs(obs_val - pred_mean),
            })
        if not stats_rows:
            stats_rows = []  # ensure DataFrame has correct columns even if empty
        stats_df = pd.DataFrame(
            stats_rows,
            columns=["COMMUNE", "REGION", "observed_T1", "predicted_mean",
                     "ci_lower", "ci_upper", "within_95ci", "abs_error"],
        )
        stats_df.to_csv(SIMULATION_DIR / "posterior_stats.csv", index=False,
                        encoding="utf-8-sig")
        print(f"  saved -> {SIMULATION_DIR.relative_to(ROOT)}/posterior_stats.csv")

    # ---- Trajectory simulations (full weekly Z per commune) -----------------
    if N_TRAJECTORIES > 0 and not SKIP_POSTERIOR_SIMS:
        n_traj = min(N_TRAJECTORIES, len(posterior_df))
        traj_idx = np.random.choice(len(posterior_df), n_traj, replace=False)

        print(f"Running {n_traj} trajectory simulations ...")

        def _run_traj(i):
            theta_i = samples[traj_idx[i]]
            p = unpack_theta(theta_i)
            beta_local = {g: p[f"beta_local_{g}"] for g in GROUP_NAMES}
            beta_intra = {g: p[f"beta_intra_{g}"] for g in GROUP_NAMES}
            wdf = run_simulation(
                beta_local=beta_local, beta_intra=beta_intra,
                beta_inter=p["beta_inter"],
                population=study["population"],
                n_patches=study["n_patches"],
                total_days=model_params["total_days"],
                week=model_params["week"],
                patch_regions=study["patch_regions"],
                initial_state=study["initial_state"],
                patch_names=study["patch_names"],
                sigma=model_params["sigma"],
                gamma=model_params["gamma"],
                mu=model_params["mu"],
                observed_90pct=study["observed_90pct"],
                k=p["k"], rho=p["rho"],
                contact_matrix=study.get("contact_matrix"),
            )
            # Retained scenario keeps only infected communes.
            sub = wdf[["COMMUNE", "REGION", "iso_week", "Z"]].copy()
            sub.insert(0, "sample_id", i)
            return sub

        traj_dfs = Parallel(n_jobs=args.workers)(
            delayed(_run_traj)(i)
            for i in tqdm(range(n_traj), desc="Trajectories")
        )
        traj_all = pd.concat(traj_dfs, ignore_index=True)
        traj_all["iso_week"] = traj_all["iso_week"].astype(int)
        traj_all.to_csv(SIMULATION_DIR / "trajectories.csv.gz", index=False,
                        compression="gzip", encoding="utf-8-sig")
        print(f"  saved -> {SIMULATION_DIR.relative_to(ROOT)}/trajectories.csv.gz")

    # ---- Save meta info -----------------------------------------------------
    meta = {
        "rounds": args.rounds,
        "num_workers": args.workers,
        "total_simulations": sum(args.rounds),
        "posterior_samples": SAMPLES,
        "elapsed_seconds": elapsed,
        "infected_sorted": infected_sorted,
        "param_names": param_names(),
        "group_names": GROUP_NAMES,
        "patches_mode": PATCHES_MODE,
        "sampler": args.sampler,
        "connectivity": CONNECTIVITY,
        "seed": SEED,
        "strategy": STRATEGY,
        "flow": FLOW,
    }
    with open(INFERENCE_DIR / "calibration_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\nDone. Results in {RESULTS_DIR.relative_to(ROOT)}/")
    print("Run  quarto render analysis.qmd  for visualisations.")


if __name__ == "__main__":
    main()
