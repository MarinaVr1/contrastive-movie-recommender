import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_model_aware_hard_infonce
from src.evaluation import evaluate_model
def reset_seed(seed=42):
    np.random.seed(seed)
def num_negatives_sweep():
    data = MovieLensData(path="data/u.data")

    negative_counts = [1, 2, 4, 8, 16]
    recalls = []

    for num_neg in negative_counts:

        reset_seed(42)
        model = MatrixFactorization(data.n_users, data.n_items)
        train_model_aware_hard_infonce(model=model,data=data,epochs=10,lr=0.01,lambda_reg=0.001,num_negatives=num_neg,tau=0.1,hard_pool_size=50)

        metrics = evaluate_model(model, data, k=10)
        recall = metrics["Recall@10"]
        recalls.append(recall)
        print(f"Num_neg={num_neg}, Recall@10={recall:.4f}")

    return negative_counts, recalls


def plot(negative_counts, recalls):

    plt.figure()
    plt.plot(negative_counts, recalls, marker='o')
    plt.xlabel("Number of Negatives per Positive")
    plt.ylabel("Recall@10")
    plt.title("Number of Negatives Analysis")
    plt.tight_layout()
    plt.savefig("results/num_negatives_sweep.png")


if __name__ == "__main__":
    negs, recalls = num_negatives_sweep()
    plot(negs, recalls)