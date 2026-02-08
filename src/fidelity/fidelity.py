import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
import matplotlib.pyplot as plt
import os

def probs(sample, bins):
    """
    Computes probabilities
    """
    pesos = np.ones_like(sample)/len(sample)
    probs, b = np.histogram(sample, weights=pesos, bins=bins)
    return probs, b

def kl_divergence(original, samples):
    p_original, _ = probs(original, bins=100)
    divergences = []
    for s in samples:
        p, _ = probs(s, bins=100)
        p[p==0.0] = 1.e-15 # evita divisao por 0
        d = sum(rel_entr(p_original, p))
        divergences.append(d)
    return divergences


def js_divergence(original, samples, col=0):
    """
    Computes the Jensen shannon distance
    Params:
        - original: original dataset 1d-array
        - samples: array of arrays with shape (n, 4)
    """
    divergences = []
    p, _ = probs(original, bins=100)
    for s in samples:
        if len(s.shape)==1:
            s = s.reshape(-1,1)
        q, _ = probs(s[:, col], 100)
        js = jensenshannon(p, q)
        divergences.append(js)
    return divergences


def w_distance(X, Y):
   
    shape_X, shape_Y = X.shape, Y.shape
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    distances = []
    for i in range(X.shape[-1]):
        r = X[:, i]
        f = Y[:, i]
        pr, _ = probs(r, 50)
        pf, _ = probs(f, 50)
        bins = np.arange(len(pr))
        wd = wasserstein_distance(bins, bins, pr, pf)
        distances.append(wd) 
    return distances


def compute_probabilities(series_real, series_fake):
    e = 1e-15

    series_real = series_real.astype(str)
    series_fake = series_fake.astype(str)

    unique = series_real._append(series_fake).unique()

    pr = (
        series_real
        .value_counts(normalize=True)
        .reindex(unique, fill_value=e)
        .values
    )

    pf = (
        series_fake
        .value_counts(normalize=True)
        .reindex(unique, fill_value=e)
        .values
    )

    return pr, pf

    
    
    
def fidelity_compute(df_real, df_fake ,categorical):
	dict_divergences = {}
	for col in df_fake.columns:
		divergences = []
		
		if col in categorical:
			pr, pf = compute_probabilities(df_real[col], df_fake[col])
			kl = sum(kl_div(pr, pf))
			js = jensenshannon(pr, pf)
			bins = np.arange(len(pr))
			wd = wasserstein_distance(bins, bins, pr, pf)
		else:
			kl = kl_divergence(df_real[col], [df_fake[col].values])[0]
			js = js_divergence(df_real[col], [df_fake[col].values], col=0)[0]
			wd = w_distance(df_real[col].values, df_fake[col].values)[0]
		divergences.append([kl, js, wd])
			
		dict_divergences[col] = np.array(divergences)
  
	return dict_divergences


def plotFidelity(divergence_dict, save_path):
    metrics = [
        "Kullback-Leibler (KL-d)",
        "Jensen-Shannon (JS-d)",
        "Wasserstein (W1)"
    ]
    columns = list(divergence_dict.keys())

    values = np.array([divergence_dict[col].flatten() for col in columns])
    values = np.clip(values, 1e-12, None)

    x = np.arange(len(columns))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(3):
        ax.bar(
            x + i * bar_width,
            values[:, i],
            width=bar_width,
            label=metrics[i]
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_ylabel("Divergence (log scale)")
    ax.set_title("Distribution Divergence per Feature")
    ax.set_yscale("log")  

    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, which="both")

    plt.tight_layout()

    save_path = os.path.join(save_path, "plots")
    os.makedirs(save_path, exist_ok=True)
    fid_path = os.path.join(save_path, "fidelity.pdf")

    fig.savefig(fid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
