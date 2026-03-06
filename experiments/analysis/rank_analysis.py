import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from src.model import MatrixFactorization
from src.data_utils import MovieLensData
from src.train import train_bpr, train_infonce
RESULTS_DIR = "results/rank_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_ranks(model, data):
    ranks = []

    for user_id, test_items in data.user_test_dict.items():
        if len(test_items) == 0:
            continue

        pos_item = test_items[0]

        scores = model.score_all(user_id)

        train_items = data.train_history[user_id]
        scores[list(train_items)] = -np.inf

        sorted_indices = np.argsort(scores)[::-1]

        rank = np.where(sorted_indices == pos_item)[0][0] + 1
        ranks.append(rank)

    return np.array(ranks)

def save_rank_cdf(rank_dict):
    plt.figure(figsize=(7, 5))

    for model_name, ranks in rank_dict.items():
        sorted_ranks = np.sort(ranks)
        y = np.arange(len(sorted_ranks)) / len(sorted_ranks)
        plt.plot(sorted_ranks, y, label=model_name)

    plt.xlabel("Rank of Positive Item")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Positive Item Rank")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rank_cdf.png"), dpi=300)
    plt.close()

def save_rank_statistics(rank_dict):

    with open(os.path.join(RESULTS_DIR, "rank_stats.txt"), "w") as f:
        f.write("===== Rank Statistics =====\n\n")

        for model_name, ranks in rank_dict.items():
            f.write(f"--- {model_name} ---\n")
            f.write(f"Mean Rank: {np.mean(ranks)}\n")
            f.write(f"Median Rank: {np.median(ranks)}\n")
            f.write(f"Top-1 Accuracy: {np.mean(ranks == 1)}\n")
            f.write(f"Top-10 Accuracy: {np.mean(ranks <= 10)}\n\n")

        model_names = list(rank_dict.keys())

        f.write("===== Pairwise T-tests =====\n\n")

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1 = model_names[i]
                m2 = model_names[j]
                t_stat, p_value = ttest_ind(rank_dict[m1], rank_dict[m2])
                f.write(f"{m1} vs {m2}\n")
                f.write(f"t-statistic: {t_stat}\n")
                f.write(f"p-value: {p_value}\n\n")
def run_experiment():
    data = MovieLensData(path='data/u.data',split_type='leave_one_out')
    rank_dict = {}
    print("Training BPR...")
    model_bpr = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
    train_bpr(model_bpr, data, epochs=25, lr=0.01)

    print("Training InfoNCE...")
    model_inf = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
    train_infonce(model_inf, data, epochs=25,
                  lr=0.01, num_negatives=4, tau=0.5)
    rank_dict["BPR"] = compute_ranks(model_bpr, data)
    rank_dict["InfoNCE"] = compute_ranks(model_inf, data)
    save_rank_cdf(rank_dict)
    save_rank_statistics(rank_dict)
    print("\nRank analysis saved in:", RESULTS_DIR)


if __name__ == "__main__":
    run_experiment()