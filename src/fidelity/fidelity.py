import numpy as np
from scipy.spatial.distance import euclidean, pdist, cdist, jensenshannon
from scipy.special import rel_entr
from scipy.stats import ks_2samp, kstest, wasserstein_distance
from scipy.special import kl_div
import matplotlib.pyplot as plt
import pandas as pd

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
    
# def js_divergence(original, samples, col=0):
#     divergences = []
#     p, _ = probs(original, bins=100)
#     for s in samples:
#         print (s.shape)
#         q, _ = probs(s[:, col], 100)
#         js = jensenshannon(p, q)
#         divergences.append(js)
#     return divergences

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

def cdf(data, bins):
    count, bins_count = np.histogram(data, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf, pdf, bins_count

def ks_teste(original, fakes, bins=50):
    """
    Returns a list of pvalues for each fake sample in fakes.

    ks_2samp p_value > 0.05 the samples comes from the same distribution
    """
    ks_testes = [ks_2samp(original, f) for f in fakes]
    return ks_testes




def w_distance(X, Y):
    """
    TODO: Compute distance considering the multivariate scenario: https://github.com/PythonOT/POT/issues/182
    """
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

def mmd(X, Y, sigma=1.0):
    """
    Implements the Maximum Mean Discrepancy. Adapted from https://torchdrift.org/notebooks/note_on_mmd.html

    """
    X = np.array(X)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1,1)
    n, d = X.shape
    m, d2 = Y.shape
    assert (d == d2)
    XY = np.concatenate([X, Y], axis=0)
    dists = cdist(XY, XY)
    k = np.exp((-1/(2*sigma**2)) * dists**2) + np.eye(n+m) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd


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


def plot(metrics, models, features):
    # Configurações
    
    
    X = np.arange(len(features)) * 1.2   # espaço entre grupos
    bar_width = 0.25
    gap_factor = 1.5                     # controla o gap entre barras

    # Offsets com gap
    offsets = np.linspace(
        -bar_width * gap_factor,
        bar_width * gap_factor,
        len(models)
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.5),
        sharey=True
    )

    for m, ax in enumerate(axes):
        for i, feature in enumerate(features):
            for k, (name, divergences, color, hatch) in enumerate(models):
                ax.bar(
                    X[i] + offsets[k],
                    divergences[feature][0][m],
                    width=bar_width,
                    color=color,
                    hatch=hatch,
                    label=name if (m == 0 and i == 0) else None
                )

        ax.set_title(metrics[m])
        ax.set_xticks(X)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    axes[0].set_ylabel('Divergence score (log scale)')

    # Legenda única
    fig.legend(
        loc='upper center',
        ncol=3,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('fidelity.pdf')