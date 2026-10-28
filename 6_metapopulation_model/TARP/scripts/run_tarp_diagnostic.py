#!/usr/bin/env python
"""Run the TARP posterior calibration diagnostic."""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_data, prepare_study_data, load_contact_matrix
from src.sbi_pipeline import make_simulator, build_prior

from sbi.diagnostics import run_tarp, check_tarp
from sbi.analysis.plot import plot_tarp

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch\.triangular_solve is deprecated.*",
)

DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "outputs" / "tarp"

NUM_REPLICATES = 1000
NUM_POSTERIOR_SAMPLES = 2000


def parse_args():
    p = argparse.ArgumentParser(description="TARP Diagnostic (sbi 0.25.0)")
    p.add_argument("--workers",      type=int, default=12)
    p.add_argument("--seed-patch",   type=str, default="Ain Chkef")
    p.add_argument("--total-days",   type=int, default=180)
    p.add_argument("--patches",      type=str, default="infected_only")
    p.add_argument("--connectivity", type=str, default="contact")
    p.add_argument("--contact-file", type=str, default="data/Commune_Contact_matrix.csv")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--strategy",     type=str, default="tsnpe")
    p.add_argument("--flow",         type=str, default="nsf")
    return p.parse_args()


def build_initial_state(study: dict, seed_patch: str) -> np.ndarray:
    n           = study["n_patches"]
    population  = study["population"]
    patch_names = study["patch_names"]
    state       = np.zeros((n, 6))
    state[:, 0] = population - 1
    try:
        idx = list(patch_names).index(seed_patch)
        state[idx, 2] = 1
    except ValueError:
        state[0, 2] = 1
    return state


def rebuild_direct_posterior(posterior, prior):
    """
    Extract the trained density estimator from the saved MCMC posterior and
    rebuild it as a DirectPosterior (sample_with='direct').

    This is the correct sbi >= 0.18.0 pattern — sample_with is a build-time
    setting, not a runtime argument to .sample().

    The density estimator (NSF flow) is unchanged; only the sampling wrapper
    is replaced.  No retraining occurs.
    """
    from sbi.inference.posteriors.direct_posterior import DirectPosterior

    density_estimator = posterior.potential_fn.posterior_estimator
    direct = DirectPosterior(
        posterior_estimator=density_estimator,
        prior=prior,
        device=str(next(density_estimator.parameters()).device),
    )
    return direct


def main():
    args   = parse_args()
    num_replicates        = NUM_REPLICATES
    num_posterior_samples = NUM_POSTERIOR_SAMPLES

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("  TARP Diagnostic  sbi=0.25.0")
    print("=" * 60)
    print(f"  replicates            : {num_replicates}")
    print(f"  posterior samples/rep : {num_posterior_samples}")
    print(f"  workers               : {args.workers}")
    print("=" * 60)

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data + simulator
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data ...")
    raw   = load_data(DATA_DIR)
    study = prepare_study_data(raw, patches_mode=args.patches)

    params_csv = pd.read_csv(DATA_DIR / "model_parameters.csv")
    model_cfg  = dict(zip(params_csv["Parameter"], params_csv["Value"]))
    model_params = {
        "sigma":      model_cfg["sigma"],
        "gamma":      model_cfg["gamma"],
        "mu":         model_cfg["mu"],
        "total_days": args.total_days,
        "week":       int(model_cfg["week"]),
    }
    study["initial_state"] = build_initial_state(study, args.seed_patch)

    if args.connectivity in ("contact", "avg_prob", "avg_contacts"):
        cpath      = ROOT / args.contact_file
        study["contact_matrix"] = load_contact_matrix(
            cpath, study["patch_names"], study["patch_regions"],
        )
    else:
        study["contact_matrix"] = None

    print("[2/5] Building simulator and prior ...")
    simulator_fn, infected_sorted, MISS_WEEK = make_simulator(study, model_params)
    prior = build_prior()

    # ------------------------------------------------------------------
    # 2. Load saved posterior
    # ------------------------------------------------------------------
    posterior_path = (
        ROOT / "outputs" / "inference" / "posterior.pkl"
    )
    posterior_direct_path = posterior_path.with_name("posterior_direct.pkl")
    if not posterior_path.exists() and not posterior_direct_path.exists():
        print(f"\nERROR: Posterior not found at {posterior_path}")
        print("Run run_calibration.py first.")
        sys.exit(1)

    if posterior_direct_path.exists():
        print(f"[3/5] Loading direct posterior from:\n      {posterior_direct_path.relative_to(ROOT)}")
        with open(posterior_direct_path, "rb") as f:
            posterior_direct = pickle.load(f)
        print("      Using saved DirectPosterior (no rebuild needed).")
    else:
        print(f"[3/5] Loading posterior from:\n      {posterior_path.relative_to(ROOT)}")
        with open(posterior_path, "rb") as f:
            posterior_mcmc = pickle.load(f)

        # Rebuild as DirectPosterior — same trained flow, no MCMC overhead
        posterior_direct = rebuild_direct_posterior(posterior_mcmc, prior)
        print("      Rebuilt as DirectPosterior (fallback for older calibration runs).")
        try:
            with open(posterior_direct_path, "wb") as f:
                pickle.dump(posterior_direct, f)
            print(f"      Cached direct posterior -> {posterior_direct_path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"      WARNING: could not cache posterior_direct.pkl ({exc})")

    # ------------------------------------------------------------------
    # 3. Simulate prior-predictive replicates
    # ------------------------------------------------------------------
    print(f"\n[4/5] Simulating {num_replicates} prior-predictive replicates ...")
    theta_tarp = prior.sample((num_replicates,))

    xs_list   = []
    report_at = max(1, num_replicates // 10)
    for i, th in enumerate(theta_tarp):
        xs_list.append(simulator_fn(th).squeeze(0))
        if (i + 1) % report_at == 0:
            print(f"      Simulated {i + 1:4d} / {num_replicates}")
    xs_tarp = torch.stack(xs_list)

    # ------------------------------------------------------------------
    # 4. Run TARP
    # ------------------------------------------------------------------
    print(f"\n[5/5] Running TARP ...")
    print(f"      ({num_replicates} replicates x {num_posterior_samples} posterior samples each)")

    ecp, alpha = run_tarp(
        theta_tarp,
        xs_tarp,
        posterior_direct,
        references=None,                   # sbi samples reference points internally
        num_posterior_samples=num_posterior_samples,
        use_batched_sampling=True,
        show_progress_bar=True,
        num_workers=args.workers,
    )

    # ------------------------------------------------------------------
    # 5. Metrics
    # check_tarp returns (atc, ks_pvalue)
    #
    #   ATC      : Area-To-Curve; ideal = 0.0
    #              > 0  -> overconfident (credible regions too narrow)
    #              < 0  -> underconfident (credible regions too wide)
    #
    #   KS p-val : H0 = ECP matches diagonal (perfect calibration)
    #              p > 0.05 -> fail to reject H0  (well-calibrated)
    #              p < 0.05 -> reject H0           (mis-calibrated, investigate)
    # ------------------------------------------------------------------
    atc, ks_pval = check_tarp(ecp, alpha)

    print("\n--- TARP Metrics ---")
    print(f"  ATC      : {atc:+.4f}  (ideal = 0.0)")
    print(f"  KS p-val : {ks_pval:.4f}  (should be > 0.05)")
    if ks_pval >= 0.05:
        print("  >> Posterior appears well-calibrated under TARP.")
    else:
        print("  >> WARNING: KS test rejects calibration — inspect ECP plot.")

    tarp_metrics = {
        "atc":                   float(atc),
        "ks_pval":               float(ks_pval),
        "num_replicates":        num_replicates,
        "num_posterior_samples": num_posterior_samples,
        "seed":                  args.seed,
        "patches":               args.patches,
        "connectivity":          args.connectivity,
        "strategy":              args.strategy,
        "flow":                  args.flow,
    }

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plot_tarp(ecp, alpha)
    ax.set_title(
        f"TARP — Expected Coverage\n"
        f"ATC = {atc:+.4f},   KS p = {ks_pval:.3f}\n"
        f"({num_replicates} replicates,  {num_posterior_samples} post. samples each)",
        fontsize=9,
    )
    plt.tight_layout()
    plot_path = out_dir / "tarp_coverage.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"\n  Saved plot    -> {plot_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # 7. Save tensors + metrics
    # ------------------------------------------------------------------
    results_path = out_dir / "tarp_results.pt"
    torch.save({"ecp": ecp, "alpha": alpha, "metrics": tarp_metrics}, results_path)
    print(f"  Saved tensors -> {results_path.relative_to(ROOT)}")
    print(f"\n=== TARP complete.  All results in {out_dir.relative_to(ROOT)} ===")


if __name__ == "__main__":
    main()