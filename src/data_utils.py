import pandas as pd
import numpy as np

class MovieLensData:
    def __init__(self, path='data/u.data', split_type='leave_one_out', test_ratio=0.2, seed=42):
        cols = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path, sep='\t', names=cols)

        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        
        self.n_users = df['user_id'].nunique()
        self.n_items = df['item_id'].nunique()

        if split_type == 'leave_one_out':
            df = df.sort_values(['user_id', 'timestamp'])
            self.test_df = df.groupby('user_id').tail(1)
            self.train_df = df.drop(self.test_df.index)
        elif split_type == 'random':
            np.random.seed(seed)
            train_rows = []
            test_rows = []
            for user, user_df in df.groupby('user_id'):
                user_df = user_df.sample(frac=1, random_state=seed)
                n_test = max(1, int(len(user_df) * test_ratio))
                test_part = user_df.iloc[:n_test]
                train_part = user_df.iloc[n_test:]
                if len(train_part) == 0:
                    train_part = test_part.iloc[:1]
                    test_part = test_part.iloc[1:]

                train_rows.append(train_part)
                test_rows.append(test_part)
            self.train_df = pd.concat(train_rows).reset_index(drop=True)
            self.test_df = pd.concat(test_rows).reset_index(drop=True)

        else:
            raise ValueError("split_type must be 'leave_one_out' or 'random'")
        

        self.user_pos_dict = (self.train_df.groupby('user_id')['item_id'].apply(list).to_dict())
        self.train_history = (self.train_df.groupby('user_id')['item_id'].apply(set).to_dict())
        self.user_test_dict = (self.test_df.groupby('user_id')['item_id'].apply(list).to_dict())

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
    def get_infonce_batches(self, num_negatives=4):
        users = []
        pos_items = []
        neg_items = []

        for row in self.train_df.itertuples():
            u = row.user_id
            i = row.item_id

            negatives = []
            while len(negatives) < num_negatives:
                j = np.random.randint(0, self.n_items)
                if j not in self.train_history[u]:
                    negatives.append(j)

            users.append(u)
            pos_items.append(i)
            neg_items.append(negatives)

        return (np.array(users), np.array(pos_items), np.array(neg_items))
    def get_inbatch_pairs(self, batch_size=128):
        interactions = self.train_df[['user_id', 'item_id']].values
        np.random.shuffle(interactions)

        for start in range(0, len(interactions), batch_size):
            batch = interactions[start:start+batch_size]
            yield batch[:, 0], batch[:, 1]

