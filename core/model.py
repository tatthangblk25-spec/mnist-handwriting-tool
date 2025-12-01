import numpy as np

class Sequential:
    def __init__(self, layers, loss):
        self.layers = layers          # Danh sách các layer theo thứ tự
        self.loss = loss              # Loss function (SoftmaxCrossEntropy)

    def forward(self, x, train=True):
        # Truyền dữ liệu lần lượt qua từng layer
        for l in self.layers:
            x = l.forward(x, train=train)
        return x

    def compute_loss_and_grad(self, logits, y):
        # Tính loss và gradient của loss
        loss = self.loss.forward(logits, y)
        grad = self.loss.backward()
        return loss, grad

    def backward(self, grad):
        # Truy gradient ngược qua từng layer (từ cuối → đầu)
        for l in reversed(self.layers):
            grad = l.backward(grad)
        return grad

    def params(self):
        # Trả về danh sách [(W, dW), (b, db), ...]
        p = []
        for l in self.layers:
            if hasattr(l, "params_and_grads"):
                p.extend(l.params_and_grads())
        return p

    def save(self, path):
        # Lưu tất cả trọng số theo thứ tự p0, p1, p2...
        state = {}
        idx = 0
        for l in self.layers:
            if hasattr(l, "params_and_grads"):
                for (W, _) in l.params_and_grads():
                    state[f"p{idx}"] = W
                    idx += 1
        np.savez(path, **state)

    def load(self, path):
        # Load trọng số theo đúng thứ tự đã lưu
        data = np.load(path, allow_pickle=True)
        idx = 0
        for l in self.layers:
            if hasattr(l, "params_and_grads"):
                for (W, _) in l.params_and_grads():
                    key = f"p{idx}"
                    if key in data:
                        W[:] = data[key]   # Gán lại vào đúng biến trọng số
                    idx += 1