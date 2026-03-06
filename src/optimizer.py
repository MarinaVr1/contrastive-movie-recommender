import numpy as np

class Adam:
    def __init__(self, shape, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, params, grads, indices):
        self.t += 1
        self.m[indices] = self.beta1 * self.m[indices] + (1 - self.beta1) * grads
        self.v[indices] = self.beta2 * self.v[indices] + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m[indices] / (1 - self.beta1 ** self.t)
        v_hat = self.v[indices] / (1 - self.beta2 ** self.t)

        params[indices] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)