"""Lead segmentation: elbow -> K-Means vs Ward hierarchical.

The target is never used to form clusters; it is only used afterwards to
describe them (conversion rate per segment).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import FIGURES, MODELS, RANDOM_STATE
from .data.loaders import load_stage2
from .data.preprocess import build_preprocessor, column_types
from .evaluate import save_json

K_RANGE = range(2, 11)
SILHOUETTE_TOLERANCE = 0.90


def main():
    X, y, info = load_stage2()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    numeric, categorical = column_types(X)

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

    # Silhouette creeps up with k, so the raw maximum landed on the edge of the
    # range. Parsimony rule: smallest k within 10% of the best silhouette —
    # the extra segments then have to earn their keep.
    best_k = int(list(K_RANGE)[int(np.argmax(silhouettes))])
    threshold = SILHOUETTE_TOLERANCE * max(silhouettes)
    best_k = next(k for k, s in zip(K_RANGE, silhouettes) if s >= threshold)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(matrix)

    # Ward's distance matrix is O(n^2): fit hierarchical on a sample.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(matrix), size=min(2000, len(matrix)), replace=False)
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(matrix[idx])
    ari = float(adjusted_rand_score(kmeans.labels_[idx], hier.labels_))

    _figures(matrix, kmeans, hier, idx, inertia, silhouettes, best_k)

    profiles = _profiles(X, y, kmeans.labels_, numeric, categorical)
    payload = {
        "dataset": info,
        "rows": int(len(matrix)),
        "k": best_k,
        "silhouette": round(float(silhouettes[best_k - K_RANGE[0]]), 4),
        "silhouette_best": round(float(max(silhouettes)), 4),
        "k_best_silhouette": int(list(K_RANGE)[int(np.argmax(silhouettes))]),
        "ari_kmeans_vs_hierarchical": round(ari, 4),
        "elbow": [{"k": k, "inertia": round(v, 2)} for k, v in zip(K_RANGE, inertia)],
        "silhouette_by_k": [{"k": k, "score": round(v, 4)} for k, v in zip(K_RANGE, silhouettes)],
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
          f"best silhouette was {max(silhouettes):.3f} at k="
          f"{list(K_RANGE)[int(np.argmax(silhouettes))]}")
    for row in profiles:
        print(f"  segment {row['cluster']}: {row['leads']} leads, "
              f"{row['conversion_rate']:.1%} converted")


def _figures(matrix, kmeans, hier, idx, inertia, silhouettes, best_k):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ks = list(K_RANGE)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, inertia, marker="o")
    ax.set(xlabel="Number of clusters (k)", ylabel="WCSS (inertia)",
           title="Elbow method")
    ax.axvline(best_k, color="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "elbow.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ks, silhouettes, marker="o")
    ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score",
           title="Silhouette by k")
    ax.axvline(best_k, color="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "silhouette.png", dpi=140)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap="tab10",
               s=7, alpha=0.7)
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"K-Means segments (k={best_k}, PCA view)")
    fig.tight_layout()
    fig.savefig(FIGURES / "kmeans_pca.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(points[idx, 0], points[idx, 1], c=hier.labels_, cmap="tab10",
               s=10, alpha=0.8)
    ax.set(xlabel="PC 1", ylabel="PC 2",
           title=f"Hierarchical segments (Ward, k={best_k}, sample)")
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_pca.png", dpi=140)
    plt.close(fig)

    sample = idx[:300]
    tree = linkage(matrix[sample], method="ward")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    dendrogram(tree, truncate_mode="lastp", p=20, no_labels=True, ax=ax)
    ax.set(title="Dendrogram (Ward, 300-lead sample, truncated)",
           xlabel="Leaves", ylabel="Distance")
    fig.tight_layout()
    fig.savefig(FIGURES / "dendrogram.png", dpi=140)
    plt.close(fig)


def _profiles(X, y, labels, numeric, categorical):
    rows = []
    profile_cols = _defining_columns(X, labels, categorical)
    for cluster in range(labels.max() + 1):
        mask = labels == cluster
        row = {
            "cluster": int(cluster + 1),
            "leads": int(mask.sum()),
            "share": round(float(mask.mean()), 3),
            "conversion_rate": round(float(y[mask].mean()), 3),
        }
        row["tier"] = ("hot" if row["conversion_rate"] >= 0.5
                       else "warm" if row["conversion_rate"] >= 0.3 else "cold")
        for col in numeric:
            row[f"mean {col}"] = round(float(X.loc[mask, col].mean()), 2)
        for col in profile_cols:
            counts = X.loc[mask, col].value_counts()
            row[col] = counts.index[0] if len(counts) else ""
        rows.append(row)
    return sorted(rows, key=lambda r: r["conversion_rate"], reverse=True)


def _defining_columns(X, labels, categorical, top=4):
    """Categorical fields whose mix differs most between segments.

    Showing the overall mode (e.g. 'India' in every segment) tells nobody
    anything, so rank by the largest gap between a segment's share and the
    overall share.
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
