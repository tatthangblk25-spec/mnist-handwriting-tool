import numpy as np
from .utils import he_init, im2col, col2im

# Lớp base cho mọi layer
class Layer:
    def forward(self, x, train=True): raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def params_and_grads(self): return []

# =====================
#  LỚP TÍCH CHẬP (Conv2D)
# =====================
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_c = in_channels
        self.out_c = out_channels

        # kernel_size có thể là int hoặc tuple
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.kh, self.kw = kh, kw

        self.stride = stride
        self.pad = padding

        # Khởi tạo trọng số bằng He-init (quan trọng cho ReLU)
        self.W = he_init((out_channels, in_channels, kh, kw))
        self.b = np.zeros((out_channels,), dtype=np.float32)
        self.cache = None

    def forward(self, x, train=True):
        self.x_shape = x.shape

        # im2col: chuyển ảnh thành ma trận để tích chập thành phép nhân ma trận
        cols, out_h, out_w = im2col(x, self.kh, self.kw, self.stride, self.pad)
        self.cols = cols

        # Trải kernel thành 2D để nhân với cols
        W_col = self.W.reshape(self.out_c, -1)

        # Tính output = cols @ W + bias
        out = cols @ W_col.T + self.b

        # Chuyển shape về (N, C, H, W)
        out = out.reshape(x.shape[0], out_h, out_w, self.out_c).transpose(0, 3, 1, 2)
        self.out_shape = (out_h, out_w)
        return out

    def backward(self, grad):
        N = grad.shape[0]

        # Chuyển grad về dạng 2D khớp với cols
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_c)

        # Gradient trọng số và bias
        dW = grad_reshaped.T @ self.cols
        db = grad_reshaped.sum(axis=0)

        # Tính gradient đầu vào
        W_col = self.W.reshape(self.out_c, -1)
        dcols = grad_reshaped @ W_col
        dx = col2im(dcols, self.x_shape, self.kh, self.kw,
                    self.out_shape[0], self.out_shape[1],
                    self.stride, self.pad)

        self.dW = dW.reshape(self.W.shape)
        self.db = db
        return dx

    def params_and_grads(self):
        return [(self.W, getattr(self, "dW", None)),
                (self.b, getattr(self, "db", None))]

# =====================
#  ReLU
# =====================
class ReLU(Layer):
    def forward(self, x, train=True):
        # mask: vị trí x > 0 (quan trọng cho backward)
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        # Chỉ truyền gradient tại những vị trí x > 0
        return grad * self.mask

# =====================
#  MaxPooling
# =====================
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kh = self.kw = kernel_size if isinstance(kernel_size, int) else kernel_size
        self.stride = stride

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1

        # reshape để dùng im2col
        x_reshaped = x.reshape(N * C, 1, H, W)
        cols, _, _ = im2col(x_reshaped, self.kh, self.kw, self.stride, 0)
        self.cols = cols

        # Lấy vị trí max index (quan trọng cho backward)
        self.arg_max = np.argmax(cols, axis=1)

        out = cols[np.arange(cols.shape[0]), self.arg_max].reshape(N, C, out_h, out_w)
        self.x_shape = x.shape
        self.out_hw = (out_h, out_w)
        return out

    def backward(self, grad):
        N, C, out_h, out_w = grad.shape

        # Khởi tạo gradient theo cols
        dcols = np.zeros_like(self.cols)

        # Gán gradient đúng vị trí max
        dcols[np.arange(dcols.shape[0]), self.arg_max] = grad.reshape(-1)

        # Chuyển ngược từ col về ảnh
        dx = col2im(dcols,
                    (N * C, 1, self.x_shape[2], self.x_shape[3]),
                    self.kh, self.kw,
                    out_h, out_w, self.stride, 0)
        return dx.reshape(self.x_shape)

# =====================
#  Flatten
# =====================
class Flatten(Layer):
    def forward(self, x, train=True):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

# =====================
#  Dense / Fully Connected
# =====================
class Dense(Layer):
    def __init__(self, in_features, out_features):
        self.W = he_init((in_features, out_features))
        self.b = np.zeros((out_features,), dtype=np.float32)

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        # Gradient của W và b
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)

        # Truy gradient về input
        return grad @ self.W.T

    def params_and_grads(self):
        return [(self.W, getattr(self, "dW", None)),
                (self.b, getattr(self, "db", None))]