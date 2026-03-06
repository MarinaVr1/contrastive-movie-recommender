from src.loss import *
def training_step_bpr(model, u_id, pos_id, neg_id, lr=0.01, lambda_reg=0.001):

    u = model.user_emb[u_id]
    pos = model.item_emb[pos_id]
    neg = model.item_emb[neg_id]

    loss, grad_x = bpr_loss(u, pos, neg)

    user_grad = grad_x * (pos - neg) + lambda_reg * u
    pos_grad = grad_x * u + lambda_reg * pos
    neg_grad = -grad_x * u + lambda_reg * neg

    model.user_emb[u_id] -= lr * user_grad
    model.item_emb[pos_id] -= lr * pos_grad
    model.item_emb[neg_id] -= lr * neg_grad

    return loss
def train_bpr(model, data, epochs=5, lr=0.01, lambda_reg=0.001):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        users, pos_items, neg_items = data.get_train_triplets(num_negatives=1)
        for u_id, p_id, n_id in zip(users, pos_items, neg_items):
            loss = training_step_bpr(model, u_id, p_id, n_id, lr, lambda_reg)
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def training_step_infonce(model, u_id, pos_id, neg_ids, lr=0.01, lambda_reg=0.001, tau=0.1):

    u = model.user_emb[u_id]
    pos = model.item_emb[pos_id]
    neg = model.item_emb[neg_ids]

    loss, probs = infonce_loss(u, pos, neg, tau)

    grad_scores = probs.copy()
    grad_scores[0] -= 1
    grad_scores /= tau

    user_grad = np.zeros_like(u)
    pos_grad = np.zeros_like(pos)
    neg_grad = np.zeros_like(neg)

    user_grad += grad_scores[0] * pos
    pos_grad += grad_scores[0] * u

    for i in range(len(neg_ids)):
        user_grad += grad_scores[i+1] * neg[i]
        neg_grad[i] += grad_scores[i+1] * u

    user_grad += lambda_reg * u
    pos_grad += lambda_reg * pos
    neg_grad += lambda_reg * neg

    model.user_emb[u_id] -= lr * user_grad
    model.item_emb[pos_id] -= lr * pos_grad
    model.item_emb[neg_ids] -= lr * neg_grad

    return loss
def train_infonce(model, data, epochs=5, lr=0.01, lambda_reg=0.001, num_negatives=4, tau=0.1):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        users, pos_items, neg_items = data.get_infonce_batches(num_negatives=num_negatives)
        for u_id, p_id, neg_ids in zip(users, pos_items, neg_items):
            loss = training_step_infonce(model, u_id, p_id, neg_ids,lr, lambda_reg, tau)
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def training_step_mse(model, u_id, i_id, rating, lr=0.01, lambda_reg=0.001):
    u = model.user_emb[u_id]
    item = model.item_emb[i_id]

    loss, grad = mse_loss(u, item, rating)

    user_grad = grad * item + lambda_reg * u
    item_grad = grad * u + lambda_reg * item

    model.user_emb[u_id] -= lr * user_grad
    model.item_emb[i_id] -= lr * item_grad

    return loss
def train_mse(model, data, epochs=5, lr=0.01, lambda_reg=0.001):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for row in data.train_df.itertuples():
            u_id = row.user_id
            i_id = row.item_id
            rating = row.rating
            loss = training_step_mse(model, u_id, i_id, rating,lr, lambda_reg)
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def train_inbatch(model, data, epochs=10, lr=0.01, lambda_reg=0.001, batch_size=128, tau=0.1):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch_users, batch_items in data.get_inbatch_pairs(batch_size):
            U = model.user_emb[batch_users]
            I = model.item_emb[batch_items]
            loss, grad_U, grad_I = inbatch_infonce_with_grad(U, I, tau)
            grad_U += lambda_reg * U
            grad_I += lambda_reg * I
            for i, u_id in enumerate(batch_users):
                model.user_emb[u_id] -= lr * grad_U[i]

            for i, item_id in enumerate(batch_items):
                model.item_emb[item_id] -= lr * grad_I[i]
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def train_model_aware_hard_infonce(model, data, epochs=5, lr=0.01, lambda_reg=0.001, num_negatives=4, tau=0.1, hard_pool_size=50):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for row in data.train_df.itertuples():
            u_id = row.user_id
            pos_id = row.item_id
            u = model.user_emb[u_id]
            scores = model.item_emb @ u
            user_history = data.train_history[u_id]
            scores[list(user_history)] = -np.inf
            hard_candidates = np.argsort(scores)[::-1][:hard_pool_size]
            if len(hard_candidates) < num_negatives:
                continue
            neg_ids = np.random.choice(hard_candidates, size=num_negatives, replace=False)
            loss = training_step_infonce(model, u_id, pos_id, neg_ids, lr, lambda_reg, tau)
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def train_symmetric_inbatch(model, data, epochs=10, lr=0.07, lambda_reg=0.001, batch_size=128, tau=0.2):
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch_users, batch_items in data.get_inbatch_pairs(batch_size):
            U = model.user_emb[batch_users]
            I = model.item_emb[batch_items]
            loss, grad_U, grad_I = symmetric_inbatch_infonce_with_grad(U, I, tau)
            grad_U += lambda_reg * U
            grad_I += lambda_reg * I
            model.user_emb[batch_users] -= lr * grad_U
            model.item_emb[batch_items] -= lr * grad_I
            total_loss += loss
            count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss/count:.4f}")
def train_curriculum_infonce(model, data,warmup_epochs=7,hard_epochs=8,lr=0.01,lambda_reg=0.001,num_negatives=4,tau=0.1,hard_pool_size=50):
    train_model_aware_hard_infonce(model, data,epochs=hard_epochs,lr=lr,lambda_reg=lambda_reg,num_negatives=num_negatives,tau=0.7,hard_pool_size=hard_pool_size)
    train_infonce(model, data,epochs=warmup_epochs,lr=lr,lambda_reg=lambda_reg,num_negatives=num_negatives,tau=0.5)
    