import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.model import MatrixFactorization
from src.train import train_bpr, train_infonce, train_model_aware_hard_infonce, train_inbatch
from src.data_utils import MovieLensData
def plot_single_pca(embeddings, title):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_.sum()

    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
    plt.title(f"{title}\nExplained variance: {explained:.2f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

from sklearn.cluster import KMeans

def plot_pca_with_clusters(model, n_clusters=5, title="Clusters"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    embeddings = model.item_emb
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(6,6))
    plt.scatter(
        emb_2d[:,0],
        emb_2d[:,1],
        c=labels,
        cmap="tab10",
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("results/pca_hard")


def compare_models_pca(models_dict, save_path=None):
    n = len(models_dict)
    plt.figure(figsize=(6 * n, 6))

    for i, (name, model) in enumerate(models_dict.items(), 1):
        plt.subplot(1, n, i)

        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(model.item_emb)
        explained = pca.explained_variance_ratio_.sum()

        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
        plt.title(f"{name}\nVar: {explained:.2f}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

def compute_margins(model, data, n_samples=5000):
    margins = []

    users = data.train_df["user_id"].values
    items = data.train_df["item_id"].values

    idx = np.random.choice(len(users), size=min(n_samples, len(users)), replace=False)

    for i in idx:
        u_id = users[i]
        pos_id = items[i]

        u = model.user_emb[u_id]
        pos = model.item_emb[pos_id]
        while True:
            neg_id = np.random.randint(0, model.item_emb.shape[0])
            if neg_id not in data.train_history[u_id]:
                break

        neg = model.item_emb[neg_id]

        score_pos = np.dot(u, pos)
        score_neg = np.dot(u, neg)

        margins.append(score_pos - score_neg)

    return np.array(margins)


def plot_margin_comparison(model_bpr, model_random, data):
    margins_bpr = compute_margins(model_bpr, data)
    margins_random = compute_margins(model_random, data)

    plt.figure(figsize=(8,6))
    plt.hist(margins_bpr, bins=50, alpha=0.6, label="BPR", density=True)
    plt.hist(margins_random, bins=50, alpha=0.6, label="Random InfoNCE", density=True)

    plt.axvline(margins_bpr.mean(), linestyle="--", label="BPR mean")
    plt.axvline(margins_random.mean(), linestyle="--", label="Random mean")

    plt.xlabel("Margin (score_pos - score_neg)")
    plt.ylabel("Density")
    plt.title("Margin Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/margins")

    print("Mean margin BPR:", margins_bpr.mean())
    print("Mean margin Random:", margins_random.mean())
def main():

    data = MovieLensData(path = 'data/u.data')

    model_bpr = MatrixFactorization(data.n_users, data.n_items)
    train_bpr(model_bpr, data, epochs=25)

    model_random = MatrixFactorization(data.n_users, data.n_items)
    train_infonce(model_random, data, epochs=25, tau=0.5)

    model_hard = MatrixFactorization(data.n_users, data.n_items)
    #train_model_aware_hard_infonce(model_hard, data, epochs=25, tau=0.7)

    model_inbatch = MatrixFactorization(data.n_users, data.n_items)
    #train_inbatch(model_inbatch, data, epochs=25, lr=0.7, batch_size=256, tau=0.2)

    models = {
        "BPR": model_bpr,
        "Random InfoNCE": model_random,
        "Hard InfoNCE": model_hard,
        "In-batch InfoNCE": model_inbatch
    }

    #compare_models_pca(models, save_path="pca_comparison.png")
    #plot_pca_with_clusters(model=model_hard)
    plot_margin_comparison(model_bpr, model_random, data)


if __name__ == "__main__":
    main()