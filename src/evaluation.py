import numpy as np
import src.analysis as au 

def evaluate_model(model, data, k=10):
    recall_hits = 0
    ndcg_total = 0
    mrr_total = 0
    total = 0

    for _, row in data.test_df.iterrows():
        user_id = row['user_id']
        test_item = row['item_id']

        scores = model.score_all(user_id)

        if user_id in data.train_history:
            scores = scores.copy()
            scores[list(data.train_history[user_id])] = -np.inf

        ranking = np.argsort(scores)[::-1]

        rank = np.where(ranking == test_item)[0][0] + 1
        mrr_total += 1.0 / rank

        top_k = ranking[:k]

        if test_item in top_k:
            recall_hits += 1
            rank_k = np.where(top_k == test_item)[0][0]
            ndcg_total += 1.0 / np.log2(rank_k + 2)

        total += 1

    return {
        f"Recall@{k}": recall_hits / total,
        f"NDCG@{k}": ndcg_total / total,
        "MRR": mrr_total / total,
    }