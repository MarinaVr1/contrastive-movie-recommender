import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_model_aware_hard_infonce, train_inbatch
from src.evaluation import evaluate_model
def reset_seed(seed=42):
    np.random.seed(seed)
def tau_sweep():
    taus = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]
    #lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    recalls = []
    for tau in taus:
        reset_seed(42)
        data = MovieLensData(path="data/u.data")
        model = MatrixFactorization(data.n_users, data.n_items)

        train_inbatch(model, data, epochs=25, lr=0.8, batch_size=256, tau=taus)

        metrics = evaluate_model(model, data, k=10)
        recall = metrics["Recall@10"]
        recalls.append(recall)
        print(f"Tau={tau}, Recall@10={recall:.4f}")
    return taus, recalls
def plot_results(taus, recalls):
    plt.figure()
    plt.plot(taus, recalls, marker='o')
    plt.xlabel("Temperature (tau)")
    plt.ylabel("Recall@10")
    plt.title("Effect of Temperature on Recall@10")
    plt.tight_layout()
    plt.savefig("results/tau_sweep_inbatch.png")
    print("Graf sačuvan u results/tau_sweep_inbatch.png")
if __name__ == "__main__":
    taus, recalls = tau_sweep()
    plot_results(taus, recalls)