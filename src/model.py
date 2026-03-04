import numpy as np

class MatrixFactorization:

    def __init__(self, n_users, n_items, emb_dim=64):
        self.user_emb = np.random.normal(0, 0.01, (n_users, emb_dim))
        self.item_emb = np.random.normal(0, 0.01, (n_items, emb_dim))
    def get_user(self, u_id):
        return self.user_emb[u_id]

    def get_item(self, i_id):
        return self.item_emb[i_id]
    def score_all(self, user_id):
        u = self.user_emb[user_id]
        return self.item_emb @ u