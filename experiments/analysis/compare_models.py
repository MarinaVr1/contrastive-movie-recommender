import matplotlib.pyplot as plt
from src.data_utils import MovieLensData
from src.model import MatrixFactorization
import numpy as np
from numpy import random
from src.train import (
    train_bpr,
    train_infonce,
    train_inbatch,
    train_mse,
    train_model_aware_hard_infonce
)
from src.evaluation import evaluate_model
from src.baseline import PopularityModel 

def reset_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
def run_all_models():
    results = {}
    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    train_mse(model, data, epochs=25)
    metrics = evaluate_model(model, data, k=10)
    results["MF + MSE"] = metrics["Recall@10"]

    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    train_bpr(model, data, epochs=25)
    metrics = evaluate_model(model, data, k=10)
    results["BPR"] = metrics["Recall@10"]

    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    train_infonce(model, data, epochs=25, tau = 0.5)
    metrics = evaluate_model(model, data, k=10)
    results["Random InfoNCE"] = metrics["Recall@10"]

    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    train_inbatch(model, data, epochs=25, lr=0.7, batch_size=256, tau=0.2)
    metrics = evaluate_model(model, data, k=10)
    results["In-batch InfoNCE"] = metrics["Recall@10"]

    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    train_model_aware_hard_infonce(model, data, epochs=25, tau = 0.7)
    metrics = evaluate_model(model, data, k=10)
    results["Model-aware Hard"] = metrics["Recall@10"]

    reset_seed()
    data = MovieLensData(path="data/u.data")
    model = PopularityModel(data)
    metrics = evaluate_model(model, data)
    results["Popularity"] = metrics["Recall@10"]

    return results


def plot_results(results):

    models = list(results.keys())
    recalls = list(results.values())

    plt.figure()
    plt.bar(models, recalls)
    plt.xticks(rotation=45)
    plt.xlabel("Model")
    plt.ylabel("Recall@10")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
if __name__ == "__main__":
    results = run_all_models()
    print("Final Recall@10 results:")
    for model, recall in results.items():
        print(f"{model}: {recall:.4f}")

    plot_results(results)