from src.data_utils import MovieLensData
from src.model import MatrixFactorization
from src.train import train_model_aware_hard_infonce
from src.evaluation import evaluate_model

data = MovieLensData(path="data/u.data", split_type='random', test_ratio=0.2)
model = MatrixFactorization(data.n_users, data.n_items)
train_model_aware_hard_infonce(model,data,epochs=10,num_negatives=4,tau=0.7,hard_pool_size=50)
metrics = evaluate_model(model, data, k=10)
print("Model-Aware Hard Negative InfoNCE:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")