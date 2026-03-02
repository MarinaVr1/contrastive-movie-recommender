import pandas as pd
import numpy as np

class MovieLensData:
    def __init__(self, path='ml-100k/u.data'):
        cols = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path, sep='\t', names=cols)

        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        
        self.n_users = df['user_id'].nunique()
        self.n_items = df['item_id'].nunique()

        df = df.sort_values(['user_id', 'timestamp'])
        self.test_df = df.groupby('user_id').tail(1)
        self.train_df = df.drop(self.test_df.index)
        self.user_pos_dict = (self.train_df.groupby('user_id')['item_id'].apply(list).to_dict())
        self.train_history = (self.train_df.groupby('user_id')['item_id'].apply(set).to_dict())

    def get_train_triplets(self, num_negatives=4):
        users, pos_items, neg_items = [], [], []

        for row in self.train_df.itertuples():
            u = row.user_id
            i = row.item_id
            
            for _ in range(num_negatives):
                while True:
                    j = np.random.randint(0, self.n_items)
                    if j not in self.train_history[u]:
                        break
                
                users.append(u)
                pos_items.append(i)
                neg_items.append(j)
                
        return np.array(users), np.array(pos_items), np.array(neg_items)
