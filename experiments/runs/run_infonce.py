from src.model import MatrixFactorization
from src.train import train_infonce
from src.data_utils import MovieLensData
from src.evaluation import evaluate_model

data = MovieLensData(path='data/u.data')

model = MatrixFactorization(
    n_users=data.n_users,
    n_items=data.n_items,
    emb_dim=64
)

train_infonce(model, data, epochs=10, lr=0.01)

metrics = evaluate_model(model, data, k=10)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")