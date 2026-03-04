from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_mse
from src.evaluation import evaluate_model

data = MovieLensData(path="data/u.data")

model = MatrixFactorization(data.n_users, data.n_items)
train_mse(model, data, epochs=10, lr=0.01, lambda_reg=0.001)
metrics = evaluate_model(model, data, k=10)
print("MF + MSE:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")