import numpy as np
import matplotlib.pyplot as plt
from src.model import MatrixFactorization
from src.data_utils import MovieLensData
from src.train import train_bpr, train_infonce
from src.evaluation import evaluate_model


def run_experiment():
    data = MovieLensData(path="data/u.data")
    negatives_list = [1, 2, 4, 8, 16]

    bpr_results = []
    infonce_results = []

    for k in negatives_list:
        print(f"\n==== Negatives: {k} ====")

        # BPR
        model_bpr = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
        train_bpr(model_bpr, data, epochs=30, lr=0.01, lambda_reg=0.001)
        metrics_bpr = evaluate_model(model_bpr, data,k=10)
        bpr_results.append(metrics_bpr["Recall@10"])

        # InfoNCE
        model_inf = MatrixFactorization(data.n_users, data.n_items, emb_dim=64)
        train_infonce(model_inf, data, epochs=30, lr=0.01,
                      lambda_reg=0.001, num_negatives=k, tau=0.07)
        metrics_inf = evaluate_model(model_inf, data,k=10)
        infonce_results.append(metrics_inf["Recall@10"])

    plt.plot(negatives_list, bpr_results, marker='o', label='BPR')
    plt.plot(negatives_list, infonce_results, marker='o', label='InfoNCE')
    plt.xlabel("Number of Negatives")
    plt.ylabel("Recall@10")
    plt.title("Scaling with Number of Negatives")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/compare_negatives.png")
    print("Sačuvan results/compare_negatives.png")
    


if __name__ == "__main__":
    run_experiment()