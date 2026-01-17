# pca_vs_lda_shift_demo.py
# A runnable, minimal demo that prints metrics + saves figures showing:
# (1) PCA tends to follow max-variance (nuisance) direction
# (2) LDA follows discriminative direction (Fisher criterion), and stays robust under covariate shift

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return v / (np.linalg.norm(v) + 1e-12)


def nearest_centroid_1d(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    """Simple 1D nearest class-mean classifier (very lightweight baseline)."""
    m0 = z_train[y_train == 0].mean()
    m1 = z_train[y_train == 1].mean()
    return (np.abs(z_test - m1) < np.abs(z_test - m0)).astype(int)


def fisher_J(X_train: np.ndarray, y_train: np.ndarray, w: np.ndarray) -> float:
    """Compute Fisher criterion J(w) = (w^T Sb w) / (w^T Sw w)."""
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    m0 = X_train[y_train == 0].mean(axis=0, keepdims=True)
    m1 = X_train[y_train == 1].mean(axis=0, keepdims=True)
    d = (m1 - m0).reshape(-1, 1)  # mean difference

    Sb = d @ d.T
    Sw = np.zeros((X_train.shape[1], X_train.shape[1]), dtype=float)
    for c, mc in [(0, m0), (1, m1)]:
        Xc = X_train[y_train == c] - mc
        Sw += Xc.T @ Xc

    num = (w.T @ Sb @ w).item()
    den = (w.T @ Sw @ w).item()
    return num / (den + 1e-12)


def get_lda_direction(lda_model: LDA, n_features: int) -> np.ndarray:
    """Robustly extract an LDA direction vector for visualization."""
    if hasattr(lda_model, "scalings_") and lda_model.scalings_ is not None:
        w = lda_model.scalings_[:, 0]
        if w.shape[0] == n_features:
            return w
    # fallback
    if hasattr(lda_model, "coef_") and lda_model.coef_ is not None and lda_model.coef_.shape[1] == n_features:
        return lda_model.coef_[0]
    raise RuntimeError("Could not extract LDA direction from this sklearn version/config.")


def main(seed: int = 0):
    os.makedirs("figs", exist_ok=True)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1) Data: x is discriminative (small variance), y is nuisance (large variance).
    #    Test-time covariate shift: nuisance variance increases further.
    # ------------------------------------------------------------
    n = 600
    mu0 = np.array([-1.0, 0.0])
    mu1 = np.array([+1.0, 0.0])

    cov_train = np.array([[0.30, 0.0],
                          [0.00, 4.00]])
    cov_test = np.array([[0.30, 0.0],
                         [0.00, 9.00]])  # shift only on nuisance dimension

    X0 = rng.multivariate_normal(mu0, cov_train, size=n // 2)
    X1 = rng.multivariate_normal(mu1, cov_train, size=n // 2)
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )

    # apply shift only to test set: resample y coordinate with larger variance (mean stays 0)
    Xte_shift = Xte.copy()
    Xte_shift[:, 1] = rng.normal(loc=0.0, scale=np.sqrt(cov_test[1, 1]), size=len(Xte_shift))

    # ------------------------------------------------------------
    # 2) Fit PCA / LDA on training data
    # ------------------------------------------------------------
    pca = PCA(n_components=1).fit(Xtr)
    lda = LDA(n_components=1).fit(Xtr, ytr)

    Ztr_pca = pca.transform(Xtr).ravel()
    Zte_pca = pca.transform(Xte_shift).ravel()

    Ztr_lda = lda.transform(Xtr).ravel()
    Zte_lda = lda.transform(Xte_shift).ravel()

    # ------------------------------------------------------------
    # 3) Classify in 1D (same simple classifier for fairness)
    # ------------------------------------------------------------
    pred_pca = nearest_centroid_1d(Ztr_pca, ytr, Zte_pca)
    pred_lda = nearest_centroid_1d(Ztr_lda, ytr, Zte_lda)
    acc_pca = float((pred_pca == yte).mean())
    acc_lda = float((pred_lda == yte).mean())

    # ------------------------------------------------------------
    # 4) Quantify: Fisher J(w) and angle between directions
    # ------------------------------------------------------------
    w_pca = pca.components_[0]
    w_lda = get_lda_direction(lda, n_features=Xtr.shape[1])

    J_pca = fisher_J(Xtr, ytr, w_pca)
    J_lda = fisher_J(Xtr, ytr, w_lda)

    cos_angle = float(np.dot(normalize(w_pca), normalize(w_lda)))
    evr = float(pca.explained_variance_ratio_[0])

    print("=== PCA vs LDA (1D) under covariate shift (test nuisance variance increased) ===")
    print(f"PCA: acc={acc_pca:.3f} | Fisher J(w)={J_pca:.4g} | explained_var_ratio={evr:.3f}")
    print(f"LDA: acc={acc_lda:.3f} | Fisher J(w)={J_lda:.4g}")
    print(f"cos(angle(w_pca, w_lda))={cos_angle:.4f}")

    # ------------------------------------------------------------
    # 5) Figure 1: 2D scatter (train + shifted test) + direction arrows
    # ------------------------------------------------------------
    plt.figure(figsize=(6.4, 5.3))

    plt.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=10, alpha=0.35, label="train class 0")
    plt.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=10, alpha=0.35, label="train class 1")
    plt.scatter(Xte_shift[yte == 0, 0], Xte_shift[yte == 0, 1], s=14, alpha=0.20, label="test-shift class 0")
    plt.scatter(Xte_shift[yte == 1, 0], Xte_shift[yte == 1, 1], s=14, alpha=0.20, label="test-shift class 1")

    # Arrow scaling based on plot extents
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    scale = 0.45 * min(x_max - x_min, y_max - y_min)

    wp = normalize(w_pca) * scale
    wl = normalize(w_lda) * scale

    # use quiver for clean arrows (no manual colors)
    plt.quiver(0, 0, wp[0], wp[1], angles="xy", scale_units="xy", scale=1)
    plt.text(wp[0], wp[1], " PCA dir", fontsize=10)

    plt.quiver(0, 0, wl[0], wl[1], angles="xy", scale_units="xy", scale=1)
    plt.text(wl[0], wl[1], " LDA dir", fontsize=10)

    plt.title("2D data + PCA/LDA directions (test has stronger nuisance)")
    plt.xlabel("x (discriminative, low variance)")
    plt.ylabel("y (nuisance, high variance)")
    plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig("figs/pca_lda_scatter.png", dpi=220)
    plt.savefig("figs/pca_lda_scatter.pdf")
    plt.close()

    # ------------------------------------------------------------
    # 6) Figure 2: 1D projection histograms (train vs test-shift; PCA vs LDA)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), sharex="col")

    bins = 35

    # PCA train
    axes[0, 0].hist(Ztr_pca[ytr == 0], bins=bins, alpha=0.6, density=True, label="class 0")
    axes[0, 0].hist(Ztr_pca[ytr == 1], bins=bins, alpha=0.6, density=True, label="class 1")
    axes[0, 0].set_title("PCA 1D (train)")
    axes[0, 0].legend(fontsize=8)

    # PCA test-shift
    axes[0, 1].hist(Zte_pca[yte == 0], bins=bins, alpha=0.6, density=True, label="class 0")
    axes[0, 1].hist(Zte_pca[yte == 1], bins=bins, alpha=0.6, density=True, label="class 1")
    axes[0, 1].set_title("PCA 1D (test-shift)")
    axes[0, 1].legend(fontsize=8)

    # LDA train
    axes[1, 0].hist(Ztr_lda[ytr == 0], bins=bins, alpha=0.6, density=True, label="class 0")
    axes[1, 0].hist(Ztr_lda[ytr == 1], bins=bins, alpha=0.6, density=True, label="class 1")
    axes[1, 0].set_title("LDA 1D (train)")
    axes[1, 0].legend(fontsize=8)

    # LDA test-shift
    axes[1, 1].hist(Zte_lda[yte == 0], bins=bins, alpha=0.6, density=True, label="class 0")
    axes[1, 1].hist(Zte_lda[yte == 1], bins=bins, alpha=0.6, density=True, label="class 1")
    axes[1, 1].set_title("LDA 1D (test-shift)")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_ylabel("density")
    axes[1, 0].set_xlabel("1D projection value")
    axes[1, 1].set_xlabel("1D projection value")

    plt.tight_layout()
    plt.savefig("figs/pca_lda_1d_hist.png", dpi=220)
    plt.savefig("figs/pca_lda_1d_hist.pdf")
    plt.close()

    print("Saved figures:")
    print(" - figs/pca_lda_scatter.(png/pdf)")
    print(" - figs/pca_lda_1d_hist.(png/pdf)")


if __name__ == "__main__":
    main(seed=0)
