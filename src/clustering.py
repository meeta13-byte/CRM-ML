"""Customer segmentation: K-Means vs Ward, with the eye and the number agreeing.

Uses only customer profile and relationship-history fields. The contact-timing
fields (day, month, contact, campaign) are dropped because they describe the
last campaign operation, not the customer, and `duration`/`y` are excluded
because they are post-outcome. The target is never used to form clusters; it is
only used afterwards to describe them.

    python3 -m src.clustering
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import FIGURES, MODELS, PALETTE, RANDOM_STATE
from .data.loaders import load_bank
from .data.preprocess import build_preprocessor
from .evaluate import save_json

K_RANGE = range(2, 11)
# Campaign-operation fields: not part of the customer portrait.
SEGMENT_EXCLUDE = ["day", "month", "contact", "campaign"]


def main():
    X, y, info = load_bank()
    X = X.drop(columns=SEGMENT_EXCLUDE).reset_index(drop=True)
    y = y.reset_index(drop=True)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    matrix = np.asarray(
        build_preprocessor(numeric, categorical).fit_transform(X), dtype=float
    )

    inertia, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(matrix)
        inertia.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(
            matrix, km.labels_, sample_size=min(3000, len(matrix)),
            random_state=RANDOM_STATE)))

    # Highest silhouette wins. The curve is flat between k=2 and k=3, and
    # both are weak (0.38), so the report should call the segments coarse
    # portraits rather than well-separated groups.
    best_k = int(list(K_RANGE)[int(np.argmax(silhouettes))])
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(matrix)

    # Ward's distance matrix is O(n^2): fit hierarchical on a sample.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(matrix), size=min(2000, len(matrix)), replace=False)
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(matrix[idx])
    ari = float(adjusted_rand_score(kmeans.labels_[idx], hier.labels_))

    _figures(matrix, kmeans, hier, idx, inertia, silhouettes, best_k)

    base_rate = float(y.mean())
    profiles = _profiles(X, y, kmeans.labels_, numeric, categorical, base_rate)
    payload = {
        "dataset": info,
        "rows": int(len(matrix)),
        "features_used": X.columns.tolist(),
        "features_excluded": SEGMENT_EXCLUDE + ["duration", "y"],
        "k": best_k,
        "silhouette": round(float(silhouettes[best_k - K_RANGE[0]]), 4),
        "silhouette_best": round(float(max(silhouettes)), 4),
        "k_choice": "highest silhouette score",
        "ari_kmeans_vs_hierarchical": round(ari, 4),
        "base_conversion_rate": round(base_rate, 4),
        "elbow": [{"k": k, "inertia": round(v, 2)} for k, v in zip(K_RANGE, inertia)],
        "silhouette_by_k": [{"k": k, "score": round(v, 4)}
                            for k, v in zip(K_RANGE, silhouettes)],
        "profiles": profiles,
        "figures": [
            "reports/figures/elbow.png",
            "reports/figures/silhouette.png",
            "reports/figures/kmeans_pca.png",
            "reports/figures/hierarchical_pca.png",
            "reports/figures/dendrogram.png",
        ],
    }
    save_json(payload, MODELS / "clusters.json")

    print(f"k = {best_k} | silhouette = {silhouettes[best_k - K_RANGE[0]]:.3f} | "
          f"ARI(K-Means, Ward) = {ari:.3f} | "
          f"max silhouette {max(silhouettes):.3f}")
    for row in profiles:
        print(f"  segment {row['cluster']}: {row['customers']} customers, "
              f"{row['conversion_rate']:.1%} subscribed ({row['tier']})")


def _figures(matrix, kmeans, hier, idx, inertia, silhouettes, best_k):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ks = list(K_RANGE)
    p = PALETTE
    plt.rcParams.update({
        "figure.facecolor": p["workspace"],
        "axes.facecolor": p["workspace"],
        "savefig.facecolor": p["workspace"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": p["text"],
        "axes.labelcolor": p["muted"],
        "axes.edgecolor": p["border"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": p["text"],
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    def clean(ax, grid_y=True):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, pad=4)
        if grid_y:
            ax.grid(axis="y", color="#F0F0F5", linewidth=0.8)
            ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, inertia, marker="o", color=p["purple_deep"], linewidth=2,
            markersize=4)
    ax.set(xlabel="Number of clusters (k)", ylabel="WCSS (inertia)",
           title="Elbow method")
    ax.axvline(best_k, color=p["border"], ls="--", lw=1)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "elbow.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, silhouettes, marker="o", color=p["purple"], linewidth=2,
            markersize=4)
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score",
           title="Silhouette by k")
    ax.axvline(best_k, color=p["border"], ls="--", lw=1)
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "silhouette.png", dpi=140)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(matrix)
    segment_cmap = LinearSegmentedColormap.from_list(
        "segments", [p["lavender"], p["purple"], p["purple_deep"]]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(points[:, 0], points[:, 1], c=kmeans.labels_,
               cmap=segment_cmap, s=7, alpha=0.75, edgecolors="none")
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"K-Means segments (k={best_k}, PCA view)")
    clean(ax, grid_y=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "kmeans_pca.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(points[idx, 0], points[idx, 1], c=hier.labels_,
               cmap=segment_cmap, s=10, alpha=0.85, edgecolors="none")
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"Hierarchical segments (Ward, k={best_k}, sample)")
    clean(ax, grid_y=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_pca.png", dpi=140)
    plt.close(fig)

    sample = idx[:300]
    tree = linkage(matrix[sample], method="ward")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    dendrogram(tree, truncate_mode="lastp", p=20, no_labels=True, ax=ax,
               link_color_func=lambda _: p["purple"], above_threshold_color=p["border"])
    ax.set(title="Dendrogram (Ward, 300-customer sample, truncated)",
           xlabel="Leaves", ylabel="Distance")
    clean(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "dendrogram.png", dpi=140)
    plt.close(fig)


def _profiles(X, y, labels, numeric, categorical, base_rate):
    rows = []
    profile_cols = _defining_columns(X, labels, categorical)
    for cluster in range(labels.max() + 1):
        mask = labels == cluster
        row = {
            "cluster": int(cluster + 1),
            "customers": int(mask.sum()),
            "share": round(float(mask.mean()), 3),
            "conversion_rate": round(float(y[mask].mean()), 3),
        }
        # Tiers relative to the cohort rate, so they work for any dataset.
        row["tier"] = ("hot" if row["conversion_rate"] >= 2 * base_rate
                       else "warm" if row["conversion_rate"] >= base_rate
                       else "cold")
        for col in numeric:
            row[f"mean {col}"] = round(float(X.loc[mask, col].mean()), 2)
        for col in profile_cols:
            counts = X.loc[mask, col].value_counts()
            row[col] = counts.index[0] if len(counts) else ""
        rows.append(row)
    return sorted(rows, key=lambda r: r["conversion_rate"], reverse=True)


def _defining_columns(X, labels, categorical, top=4):
    """Categorical fields whose mix differs most between segments.

    Showing the overall mode in every segment tells nobody anything, so rank by
    the largest gap between a segment's share and the overall share.
    """
    scored = []
    for col in categorical:
        if X[col].nunique() > 40:
            continue
        overall = X[col].value_counts(normalize=True)
        gap = max(
            float((X.loc[labels == c, col].value_counts(normalize=True) - overall)
                  .abs().max())
            for c in range(labels.max() + 1)
        )
        scored.append((gap, col))
    return [col for _, col in sorted(scored, reverse=True)[:top]]


if __name__ == "__main__":
    main()
