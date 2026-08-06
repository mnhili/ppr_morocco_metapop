import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Provide standard Matplotlib parameters for dark black text and good sizes
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

def replot_publication_tarp():
    # Setup paths (flattened single-scenario layout)
    results_file = Path("outputs/tarp/tarp_results.pt")
    if not results_file.exists():
        print(f"Error: Could not find {results_file}")
        return

    out_dir = Path("outputs/tarp")
    file_prefix = out_dir / "tarp_coverage_publication"

    # Load data
    data = torch.load(results_file, map_location='cpu', weights_only=False)
    ecp = data['ecp'].numpy()
    alpha = data['alpha'].numpy()
    
    metrics = data.get('metrics', {})
    atc = metrics.get('atc', 0.0)
    ks_pval = metrics.get('ks_pval', 0.0)
    num_replicates = metrics.get('num_replicates', 1000)
    num_posterior_samples = metrics.get('num_posterior_samples', 2000)

    import scipy.stats as stats

    # Replot with specific custom colors config
    fig, ax = plt.subplots(figsize=(6, 6))

    # Reference / Ideal Coverage (pure red)
    ax.plot([0, 1], [0, 1], color='#fa0202', linestyle='--', linewidth=2, label='Ideal')

    # Add 95% HDI around the Empirical coverage line
    # Using the exact Binomial interval based on empirical counts
    N = int(num_replicates)
    lower_bound, upper_bound = stats.binom.interval(0.95, N, ecp)
    lower_bound = lower_bound / N
    upper_bound = upper_bound / N
    
    # Fill the 95% HDI region
    ax.fill_between(alpha, lower_bound, upper_bound, color='#0077b6', alpha=0.2, edgecolor='none', label='95% HDI')

    # Empirical Coverage (blue)
    ax.plot(alpha, ecp, color='#0077b6', linestyle='-', linewidth=2.5, label='TARP')

    # Figure formatting
    ax.set_xlabel(r"Credibility Level ($\alpha$)")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle=':', alpha=0.5)

    # Use EXACTLY the same text from the original title, but placed inside the figure
    text_str = (
        f"ATC = {atc:+.2f},   KS p = {ks_pval:.2f}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9)
    # Position in the top left
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, color='black', weight='bold')

    # Legend
    legend = ax.legend(loc='lower right', frameon=True, edgecolor='black')
    legend.get_frame().set_facecolor('white')

    # Export saving (tight bbox, 600 DPI, specified formats)
    save_kwargs = {'dpi': 600, 'bbox_inches': 'tight'}

    fig.savefig(f"{file_prefix}.png", **save_kwargs)
    fig.savefig(f"{file_prefix}.pdf", format="pdf", **save_kwargs)
    fig.savefig(f"{file_prefix}.eps", format="eps", **save_kwargs)
    
    try:
        fig.savefig(f"{file_prefix}.tiff", format="tiff", pil_kwargs={"compression": "tiff_lzw"}, **save_kwargs)
    except Exception as e:
        print(f"TIFF saving warning: {e}")
        
    plt.close(fig)
    print(f"TARP replots generated at {out_dir}")

if __name__ == "__main__":
    replot_publication_tarp()
