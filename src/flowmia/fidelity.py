"""
Fidelity evaluation for synthetic tabular data.

Measures per-feature distributional similarity between a real dataset and a
synthetic one using three complementary divergence metrics:

- **KL divergence** — asymmetric, sensitive to regions where the synthetic
  distribution assigns near-zero mass.
- **Jensen-Shannon divergence** — symmetric, bounded in [0, 1].
- **Wasserstein-1 distance** — earth mover's distance, accounts for the
  geometry of the distribution support.

Categorical columns use probability mass functions over observed categories.
Numerical columns use histogram-based density estimates (100 bins).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div, rel_entr
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _histogram_probs(sample: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized histogram (probability weights) for a 1-D array.

    Args:
        sample: 1-D numerical array.
        bins: Number of histogram bins.

    Returns:
        Tuple of (probabilities, bin_edges).
    """
    weights = np.ones_like(sample) / len(sample)
    probs, edges = np.histogram(sample, weights=weights, bins=bins)
    return probs, edges


def _kl_divergence(real: np.ndarray, synth: np.ndarray, bins: int = 100) -> float:
    """
    KL divergence D_KL(real || synth) from histogram estimates.

    Args:
        real: 1-D array of real values.
        synth: 1-D array of synthetic values.
        bins: Number of histogram bins.

    Returns:
        Scalar KL divergence.
    """
    p, _ = _histogram_probs(real, bins)
    q, _ = _histogram_probs(synth, bins)
    q[q == 0.0] = 1e-15  # avoid log(0)
    return float(np.sum(rel_entr(p, q)))


def _js_divergence(real: np.ndarray, synth: np.ndarray, bins: int = 100) -> float:
    """
    Jensen-Shannon divergence between histogram estimates of real and synth.

    Args:
        real: 1-D array of real values.
        synth: 1-D array of synthetic values.
        bins: Number of histogram bins.

    Returns:
        Scalar JS divergence in [0, 1].
    """
    p, _ = _histogram_probs(real, bins)
    q, _ = _histogram_probs(synth, bins)
    return float(jensenshannon(p, q))


def _wasserstein(real: np.ndarray, synth: np.ndarray, bins: int = 50) -> float:
    """
    Wasserstein-1 distance between histogram estimates of real and synth.

    Uses bin indices as the metric space so that the distance reflects the
    number of histogram "steps" needed to transport mass.

    Args:
        real: 1-D array of real values.
        synth: 1-D array of synthetic values.
        bins: Number of histogram bins.

    Returns:
        Scalar Wasserstein distance.
    """
    p, _ = _histogram_probs(real.reshape(-1), bins)
    q, _ = _histogram_probs(synth.reshape(-1), bins)
    bin_indices = np.arange(len(p))
    return float(wasserstein_distance(bin_indices, bin_indices, p, q))


def _categorical_probs(real, synth) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute aligned probability mass functions for two categorical series.

    The union of all observed categories is used as the support so that
    both distributions are defined over the same domain.

    Args:
        real: pandas Series of real categorical values.
        synth: pandas Series of synthetic categorical values.

    Returns:
        Tuple of (p_real, p_synth) as numpy arrays over the shared support.
    """
    eps = 1e-15
    real = real.astype(str)
    synth = synth.astype(str)
    unique = real._append(synth).unique()

    p = real.value_counts(normalize=True).reindex(unique, fill_value=eps).values
    q = synth.value_counts(normalize=True).reindex(unique, fill_value=eps).values
    return p, q


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fidelity_compute(df_real, df_fake, categorical: list) -> dict:
    """
    Compute per-feature distributional divergences between real and synthetic data.

    For each column in ``df_fake``:
    - Categorical columns use PMF-based KL, JS, and Wasserstein estimates.
    - Numerical columns use histogram-based estimates.

    Args:
        df_real: Real (member) dataframe.
        df_fake: Synthetic dataframe with the same columns.
        categorical: List of column names to treat as categorical.

    Returns:
        Dictionary mapping each column name to a (1, 3) numpy array
        containing ``[kl_divergence, js_divergence, wasserstein_distance]``.
    """
    divergences = {}

    for col in df_fake.columns:
        if col in categorical:
            p, q = _categorical_probs(df_real[col], df_fake[col])
            kl = float(np.sum(kl_div(p, q)))
            js = float(jensenshannon(p, q))
            bin_indices = np.arange(len(p))
            wd = float(wasserstein_distance(bin_indices, bin_indices, p, q))
        else:
            real_vals = df_real[col].values
            fake_vals = df_fake[col].values
            kl = _kl_divergence(real_vals, fake_vals)
            js = _js_divergence(real_vals, fake_vals)
            wd = _wasserstein(real_vals, fake_vals)

        divergences[col] = np.array([[kl, js, wd]])

    return divergences


def plotFidelity(divergence_dict: dict, save_path: str) -> None:
    """
    Save a grouped bar chart of per-feature divergences (log scale).

    Three bars are drawn for each feature: KL divergence, JS divergence,
    and Wasserstein distance. The y-axis uses a log scale to handle the
    wide range of values typical across features.

    Args:
        divergence_dict: Output of :func:`fidelity_compute`.
        save_path: Root directory; the plot is written to
            ``save_path/plots/fidelity.pdf``.
    """
    metric_labels = [
        "Kullback-Leibler (KL)",
        "Jensen-Shannon (JS)",
        "Wasserstein (W1)",
    ]
    columns = list(divergence_dict.keys())
    values = np.array([divergence_dict[col].flatten() for col in columns])
    values = np.clip(values, 1e-12, None)

    x = np.arange(len(columns))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, label in enumerate(metric_labels):
        ax.bar(x + i * bar_width, values[:, i], width=bar_width, label=label)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_ylabel("Divergence (log scale)")
    ax.set_title("Distribution Divergence per Feature")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, which="both")

    plt.tight_layout()

    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "fidelity.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")