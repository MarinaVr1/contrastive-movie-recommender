from src.utils import sigmoid
import numpy as np
def bpr_loss(user, pos_item, neg_item):
    s_pos = user @ pos_item
    s_neg = user @ neg_item
    x = s_pos - s_neg
    sig = sigmoid(x)
    loss = -np.log(sig + 1e-10)
    grad_x = sig - 1
    return loss, grad_x
def infonce_loss(user, pos_item, neg_item, tau=0.1):
    s_pos = user @ pos_item
    s_neg = neg_item @ user
    scores = np.concatenate(([s_pos], s_neg)) / tau
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    loss = -np.log(probs[0] + 1e-10)
    return loss, probs
def mse_loss(user, item, rating):
    pred = user @ item
    error = pred - rating
    loss = error ** 2
    grad = 2 * error
    return loss, grad
def inbatch_infonce_with_grad(users, items, tau=0.1):
    B = users.shape[0]
    users = users / (np.linalg.norm(users, axis=1, keepdims=True) + 1e-10)
    items = items / (np.linalg.norm(items, axis=1, keepdims=True) + 1e-10)
    S = users @ items.T
    S /= tau
    S -= np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    loss = -np.mean(np.log(np.diag(P) + 1e-10))
    grad_S = P.copy()
    grad_S[np.arange(B), np.arange(B)] -= 1
    grad_S /= B
    grad_users = grad_S @ items / tau
    grad_items = grad_S.T @ users / tau
    return loss, grad_users, grad_items
