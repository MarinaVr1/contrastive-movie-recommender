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
