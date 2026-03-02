import numpy as np
from collections import Counter
class PopularityModel:
    def __init__(self, data):
        self.n_items = data.n_items
        item_counts = Counter(data.train_df['item_id'])
        self.popularity = np.zeros(self.n_items)
        for item_id, count in item_counts.items():
            self.popularity[item_id] = count

    def recommend(self, user_history, k=10):
        scores = self.popularity.copy()
        scores[list(user_history)] = -np.inf
        top_k = np.argsort(-scores)[:k]
        return top_k
    def score_all(self, user_id):
        return self.popularity