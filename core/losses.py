import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y = None

    def forward(self, logits, y):
        # Trừ max để tránh overflow khi exp
        logits = logits - logits.max(axis=1, keepdims=True)

        # Softmax
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.probs = probs

        # Lưu nhãn thật
        self.y = y.astype(int)

        # Tính loss = -log p
        N = y.shape[0]
        loss = -np.log(probs[np.arange(N), self.y] + 1e-12).mean()
        return loss

    def backward(self):
        N = self.y.shape[0]

        # grad = softmax - one_hot
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1.0
        grad /= N  # chia batch size
        return grad