import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_bpr, train_infonce, train_model_aware_hard_infonce, train_inbatch
from src.evaluation import evaluate_model


def reset_seed(seed=42):
    np.random.seed(seed)
def epoch_stability():
    reset_seed(42)
    data = MovieLensData(path="data/u.data")
    model = MatrixFactorization(data.n_users, data.n_items)
    epochs = 50
    recalls = []

    for epoch in range(17):
        #train_bpr(model, data, epochs=1, lr=0.01)
        #train_infonce(model,data,epochs=1,tau=0.5,num_negatives=4)
        train_model_aware_hard_infonce(model,data,epochs=1,num_negatives=4,tau=0.7,hard_pool_size=50)
        #train_inbatch(model, data, epochs=1, lr=0.7, batch_size=256, tau=0.2)
        metrics = evaluate_model(model, data, k=10)
        recall = metrics["Recall@10"]
        recalls.append(recall)

        print(f"Epoch {epoch+1}, Recall@10={recall:.4f}")
    for ep in range(33):
        train_infonce(model,data,epochs=1,tau=0.5,num_negatives=4)
        metrics = evaluate_model(model, data, k=10)
        recall = metrics["Recall@10"]
        recalls.append(recall)

        print(f"Epoch {epoch+1}, Recall@10={recall:.4f}")
    return recalls

def plot(recalls):

    plt.figure()
    plt.plot(range(1, len(recalls)+1), recalls, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Recall@10")
    plt.title("Epoch vs Recall@10 (Hard + Random InfoNCE)")
    plt.tight_layout()
    plt.savefig("results/epoch_stability_curriculum.png")
if __name__ == "__main__":
    recalls = epoch_stability()
    plot(recalls)