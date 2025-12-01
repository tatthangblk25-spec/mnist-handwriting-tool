import numpy as np
from PIL import Image, ImageDraw


def gen_shapes_dataset(n_samples=6000, img_size=32, noise=0.05, seed=123, bbox=False):
    """
    Generate images containing either rectangles or circles.
    Output:
      X: (N,1,H,W) ảnh grayscale đã chuẩn hóa
      y: nhãn (0=rectangle, 1=circle)
      B: bounding box chuẩn hóa (tùy chọn)
    """

    rng = np.random.default_rng(seed)  # Bộ sinh số ngẫu nhiên cố định seed

    # Tạo mảng chứa ảnh và nhãn
    X = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    B = []  # Lưu bounding box nếu cần

    for i in range(n_samples):
        # Tạo ảnh trắng đen (L = grayscale) nền đen
        img = Image.new("L", (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)

        # Chọn random: 0=rectangle, 1=circle
        shape = rng.integers(0, 2)

        # padding để không vẽ sát mép
        pad = img_size // 8

        # Random vị trí + kích thước
        x0 = rng.integers(pad, img_size - pad * 2)
        y0 = rng.integers(pad, img_size - pad * 2)
        x1 = rng.integers(x0 + pad, min(img_size - pad, x0 + img_size // 2))
        y1 = rng.integers(y0 + pad, min(img_size - pad, y0 + img_size // 2))

        # =====================
        # VẼ HÌNH CHỮ NHẬT
        # =====================
        if shape == 0:
            draw.rectangle([x0, y0, x1, y1], outline=255, width=2)
            y[i] = 0

            if bbox:
                # Tính bounding box normalized
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                w = x1 - x0
                h = y1 - y0
                B.append([cx / img_size, cy / img_size, w / img_size, h / img_size])

        # =====================
        # VẼ HÌNH TRÒN
        # =====================
        else:
            # Bán kính dựa trên kích thước x,y tạo ra
            r = max(3, int(((x1 - x0) + (y1 - y0)) / 4))
            draw.ellipse([x0, y0, x0 + 2 * r, y0 + 2 * r], fill=255)
            y[i] = 1

            if bbox:
                cx = x0 + r
                cy = y0 + r
                B.append([cx / img_size, cy / img_size, r / img_size])

        # Chuyển sang numpy array + chuẩn hóa [0,1]
        arr = (np.array(img, dtype=np.float32) / 255.0)

        # Thêm nhiễu Gaussian nhẹ
        if noise > 0:
            arr = np.clip(arr + rng.normal(0, noise, size=arr.shape), 0, 1)

        X[i, 0] = arr

    # Chuyển B thành numpy hoặc None
    B = np.array(B, dtype=np.float32) if bbox else None

    return X, y, B