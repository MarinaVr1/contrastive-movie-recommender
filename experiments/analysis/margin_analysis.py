import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

from src.model import MatrixFactorization
from src.data_utils import MovieLensData
from src.train import train_bpr, train_infonce
def compute_margins(model, data):
    margins = []

    for user_id, test_items in data.user_test_dict.items():
        if len(test_items) == 0:
            continue

        pos_item = test_items[0]

        scores = model.score_all(user_id)

        # mask training interactions
        train_items = data.train_history[user_id]
        scores[list(train_items)] = -np.inf

        s_pos = scores[pos_item]
        max_neg = np.max(scores)

        margin = s_pos - max_neg
        margins.append(margin)

    return np.array(margins)
def plot_cdf(margins_bpr, margins_inf):
    sorted_bpr = np.sort(margins_bpr)
    sorted_inf = np.sort(margins_inf)

    y_bpr = np.arange(len(sorted_bpr)) / len(sorted_bpr)
    y_inf = np.arange(len(sorted_inf)) / len(sorted_inf)

    plt.figure(figsize=(7, 5))
    plt.plot(sorted_bpr, y_bpr, label="BPR")
    plt.plot(sorted_inf, y_inf, label="InfoNCE")
    plt.xlabel("Margin (s_pos - max_neg)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Margin Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cdf.png")
    plt.close()

def plot_violin(margins_bpr, margins_inf):
    df = pd.DataFrame({
        "Margin": np.concatenate([margins_bpr, margins_inf]),
        "Model": ["BPR"] * len(margins_bpr) + ["InfoNCE"] * len(margins_inf)
    })

    plt.figure(figsize=(6, 5))
    sns.violinplot(data=df, x="Model", y="Margin")
    plt.title("Margin Distribution Comparison")
    plt.tight_layout()
    plt.savefig("results/violin.png")
    plt.close()

def statistical_summary(margins_bpr, margins_inf):
    print("\n===== Margin Statistics =====")
    print("Mean Margin BPR:", np.mean(margins_bpr))
    print("Mean Margin InfoNCE:", np.mean(margins_inf))
    print("Median Margin BPR:", np.median(margins_bpr))
    print("Median Margin InfoNCE:", np.median(margins_inf))

    print("\n% Positive Margins (s_pos > max_neg)")
    print("BPR:", np.mean(margins_bpr > 0))
    print("InfoNCE:", np.mean(margins_inf > 0))

    t_stat, p_value = ttest_ind(margins_bpr, margins_inf)
    print("\nT-test Results")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

def run_experiment():
    data = MovieLensData(path='data/u.data')
    model_bpr = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
    train_bpr(model_bpr, data, epochs=30, lr=0.01)

    margins_bpr = compute_margins(model_bpr, data)
    model_inf = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
    train_infonce(model_inf, data, epochs=30,
                  lr=0.01, num_negatives=4, tau=0.07)

    margins_inf = compute_margins(model_inf, data)

    plot_cdf(margins_bpr, margins_inf)
    plot_violin(margins_bpr, margins_inf)

    statistical_summary(margins_bpr, margins_inf)


if __name__ == "__main__":
    run_experiment()
