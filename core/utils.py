import numpy as np

# Khởi tạo He (thường dùng cho ReLU)
def he_init(shape, fan_in=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(1234)
    if fan_in is None:
        if len(shape) == 2:            # Dense layer
            fan_in = shape[0]
        elif len(shape) == 4:          # Conv layer
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = np.prod(shape[:-1])
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=shape).astype(np.float32)

# Độ chính xác phân loại
def accuracy(pred, y):
    return (pred.argmax(axis=1) == y).mean()

# One-hot encoding
def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

# Sinh batch dữ liệu
def batch_iter(X, y, batch_size, shuffle=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i+batch_size]
        yield X[sel], y[sel]

# Chuyển ảnh sang dạng cột (dùng cho Conv)
def im2col(X, kernel_h, kernel_w, stride=1, pad=0):
    N, C, H, W = X.shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    X_padded = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=X.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = X_padded[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.transpose(0,4,5,1,2,3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w

# Chuyển ngược từ col → ảnh
def col2im(cols, X_shape, kernel_h, kernel_w, out_h, out_w, stride=1, pad=0):
    N, C, H, W = X_shape
    X_padded = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=cols.dtype)

    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0,3,4,5,1,2)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            X_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]

    return X_padded[:, :, pad:H+pad, pad:W+pad]