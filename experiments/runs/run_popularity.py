from src.data_utils import MovieLensData
from src.baseline import PopularityModel
from src.evaluation import evaluate_model 

data = MovieLensData(path="data/u.data")

model = PopularityModel(data)

results = evaluate_model(model, data)

print("Popularity baseline:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")