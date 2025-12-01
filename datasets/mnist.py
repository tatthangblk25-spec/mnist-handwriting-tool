import os, struct
import numpy as np


def _read_idx_images(path):
    with open(path, 'rb') as f:
        # Đọc header IDX: magic number, số ảnh, số dòng, số cột
        # magic = 2051 cho file ảnh MNIST
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic for images")

        # Đọc toàn bộ pixel còn lại
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape thành (N, 1, H, W) và chuẩn hóa về [0,1]
        data = data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0
    return data


def _read_idx_labels(path):
    with open(path, 'rb') as f:
        # Đọc header IDX: magic number, số label
        # magic = 2049 cho file label MNIST
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic for labels")

        # Đọc toàn bộ label còn lại
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(root="data/mnist"):
    # Xác định đường dẫn các file MNIST chuẩn IDX
    paths = {
        "train_images": os.path.join(root, "train-images.idx3-ubyte"),
        "train_labels": os.path.join(root, "train-labels.idx1-ubyte"),
        "test_images":  os.path.join(root, "t10k-images.idx3-ubyte"),
        "test_labels":  os.path.join(root, "t10k-labels.idx1-ubyte"),
    }

    # Đọc dữ liệu train/test
    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test  = _read_idx_images(paths["test_images"])
    y_test  = _read_idx_labels(paths["test_labels"])

    # Trả về tuple giống kiểu MNIST trong PyTorch
    return (X_train, y_train), (X_test, y_test)
