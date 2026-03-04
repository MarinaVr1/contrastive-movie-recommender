from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_symmetric_inbatch
from src.evaluation import evaluate_model

data = MovieLensData(path="data/u.data")
model = MatrixFactorization(data.n_users, data.n_items)
train_symmetric_inbatch(model, data, epochs=10, tau=0.1)
metrics = evaluate_model(model, data, k=10)
print("Symmetric In-Batch InfoNCE:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")