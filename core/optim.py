class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = params            # Hàm hoặc list trả về (W, dW)
        self.lr = lr                    # Learning rate
        self.momentum = momentum        # Momentum
        self.weight_decay = weight_decay  # L2 regularization
        self.v = {}                     # Lưu vận tốc cho momentum

    def step(self):
        # Lấy danh sách tham số (W, dW)
        param_list = self.params() if callable(self.params) else self.params

        for (W, dW) in param_list:
            if dW is None:
                continue  # Layer không có gradient

            # L2 regularization
            if self.weight_decay != 0.0:
                dW = dW + self.weight_decay * W

            key = id(W)  # ID duy nhất để lưu momentum

            if self.momentum > 0:
                # Cập nhật theo momentum
                v = self.v.get(key, 0)
                v = self.momentum * v - self.lr * dW
                W += v
                self.v[key] = v
            else:
                # SGD thường
                W -= self.lr * dW