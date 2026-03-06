from src.model import MatrixFactorization
from src.train import train_curriculum_infonce
from src.data_utils import MovieLensData
from src.evaluation import evaluate_model

data = MovieLensData(path='data/u.data', split_type='leave_one_out')
model = MatrixFactorization(n_users=data.n_users,n_items=data.n_items,emb_dim=64)
train_curriculum_infonce(model, data,
                              warmup_epochs=9,
                              hard_epochs=11,
                              lr=0.01,
                              lambda_reg=0.001,
                              num_negatives=4,
                              tau=0.1,
                              hard_pool_size=50)
metrics = evaluate_model(model, data, k=10)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")