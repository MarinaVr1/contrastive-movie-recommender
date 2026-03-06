from src.data_utils import MovieLensData

batch_size = 128

data = MovieLensData(path="data/u.data")

false_negative_count = 0
total_checks = 0

for batch_users, batch_items in data.get_inbatch_pairs(batch_size):

    for idx in range(len(batch_users)):
        item = batch_items[idx]

        for j in range(len(batch_users)):
            if j != idx:
                other_user = batch_users[j]

                if item in data.train_history[other_user]:
                    false_negative_count += 1

                total_checks += 1

false_negative_rate = false_negative_count / total_checks

print("False negative rate:", false_negative_rate)