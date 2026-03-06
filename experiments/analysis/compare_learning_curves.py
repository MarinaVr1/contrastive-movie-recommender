import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import (
    train_infonce,
    train_model_aware_hard_infonce
)
from src.evaluation import evaluate_model
def reset_seed(seed=42):
    np.random.seed(seed)
def train_and_track_random(data, epochs=35):
    reset_seed(42)
    model = MatrixFactorization(data.n_users, data.n_items)
    recalls = []
    for epoch in range(epochs):

        train_infonce(model,data,epochs=1,tau=0.5,num_negatives=4)

        metrics = evaluate_model(model,data,k=10)
        recall = metrics["Recall@10"]
        recalls.append(recall)
        print(f"Epoch {epoch+1}, Recall@10={recall:.4f}")
    return recalls
def train_and_track_hard(data, epochs=20):
    reset_seed(42)
    model = MatrixFactorization(data.n_users, data.n_items)
    recalls = []
    for epoch in range(epochs):

        train_model_aware_hard_infonce(model,data,epochs=1,num_negatives=4,tau=0.7,hard_pool_size=50)

        metrics = evaluate_model(model,data,k=10)

        recall = metrics["Recall@10"]
        recalls.append(recall)

        print(f"Epoch {epoch+1}, Recall@10={recall:.4f}")
    return recalls

def plot(random_recalls, hard_recalls):

    epochs = range(1, len(random_recalls)+1)

    plt.figure()
    plt.plot(epochs, random_recalls, marker='o', label="Random InfoNCE")
    plt.plot(epochs, hard_recalls, marker='o', label="Hard InfoNCE")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@10")
    plt.title("Learning Curves Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/compare_learning_curves_35e.png")
    print("Sačuvan results/compare_learning_curves_35e.png")


if __name__ == "__main__":

    data = MovieLensData(path="data/u.data")
    random_recalls = train_and_track_random(data, epochs=35)
    hard_recalls = train_and_track_hard(data, epochs=35)
    
    plot(random_recalls, hard_recalls)