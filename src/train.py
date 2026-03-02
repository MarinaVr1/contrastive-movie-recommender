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
