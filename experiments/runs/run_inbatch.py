from src.model import MatrixFactorization
from src.train import train_inbatch
from src.data_utils import MovieLensData
from src.evaluation import evaluate_model

data = MovieLensData(path='data/u.data', split_type='random')

model = MatrixFactorization(n_users=data.n_users,n_items=data.n_items,emb_dim=64)
train_inbatch(model, data, epochs=25, lr=0.8, batch_size=256, tau=0.2)
metrics = evaluate_model(model, data, k=10)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")