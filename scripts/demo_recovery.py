"""
Demonstration: structure recovery via spectral (Fiedler vector) ordering.

This script visualizes the spectral clustering concept using three matrices:
1. original.png  - clean block-structured adjacency
2. shuffled.png  - vertices randomly permuted (structure hidden)
3. recovered.png - reordered by spectral Fiedler vector (structure restored)

Run:
    python scripts/demo_recovery.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from graph_models import generate_multi_group_laplacian
from SCOPE import eigen_decomposition

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True, parents=True)

def laplacian_to_adjacency(L):
    L = np.asarray(L, dtype=float)
    D = np.diag(np.diag(L))
    A = D - L
    np.fill_diagonal(A, 0.0)
    return A

def fiedler_order(L):
    """Return vertex order sorted by the Fiedler vector."""
    v = eigen_decomposition(L, gpu=False, sparse=False, k=2)
    fiedler = np.asarray(v).flatten()
    return np.argsort(fiedler)

def show_matrix(A, title, path):
    plt.figure(figsize=(5,5))
    plt.imshow(A, cmap='binary', interpolation='none')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    n_groups, group_size = 3, 30
    prob_within, prob_between = 0.9, 0.03

    # Step 1: original
    L = generate_multi_group_laplacian(n_groups, group_size, prob_within, prob_between)
    A = laplacian_to_adjacency(L)
    show_matrix(A, "Original (block structure)", FIGS / "original.png")

    # Step 2: shuffled
    perm = np.random.permutation(A.shape[0])
    A_shuf = A[perm][:, perm]
    show_matrix(A_shuf, "Shuffled (structure hidden)", FIGS / "shuffled.png")

    # Step 3: recovered
    L_shuf = np.diag(A_shuf.sum(axis=1)) - A_shuf
    order = fiedler_order(L_shuf)
    A_rec = A_shuf[order][:, order]
    show_matrix(A_rec, "Recovered (spectral order)", FIGS / "recovered.png")

    print("Saved figures to:", FIGS)

if __name__ == "__main__":
    main()