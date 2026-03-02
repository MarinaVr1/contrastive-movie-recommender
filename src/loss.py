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
