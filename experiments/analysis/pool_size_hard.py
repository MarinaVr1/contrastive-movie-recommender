import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_model_aware_hard_infonce
from src.evaluation import evaluate_model


def reset_seed(seed=42):
    np.random.seed(seed)


def hard_pool_sweep():
    pool_sizes = [10, 20, 50, 100, 200]
    recalls = []

    for pool_size in pool_sizes:

        print(f"\nRunning hard_pool_size = {pool_size}")

        reset_seed(42)

        data = MovieLensData(path="data/u.data")
        model = MatrixFactorization(data.n_users, data.n_items)

        train_model_aware_hard_infonce(model,data,epochs=10,lr=0.01,lambda_reg=0.001,num_negatives=4,tau=0.7,hard_pool_size=pool_size)

        metrics = evaluate_model(model,data,k=10)

        recall = metrics["Recall@10"]
        recalls.append(recall)

        print(f"Pool size = {pool_size}, Recall@10 = {recall:.4f}")

    return pool_sizes, recalls


def plot_results(pool_sizes, recalls):

    plt.figure()
    plt.plot(pool_sizes, recalls, marker='o')
    plt.xlabel("Hard Pool Size")
    plt.ylabel("Recall@10")
    plt.title("Effect of Hard Negative Pool Size")
    plt.tight_layout()
    plt.savefig("results/hard_pool_sweep.png")


if __name__ == "__main__":
    pool_sizes, recalls = hard_pool_sweep()
    plot_results(pool_sizes, recalls)