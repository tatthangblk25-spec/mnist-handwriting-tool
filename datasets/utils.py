import numpy as np
from PIL import Image

def to_chw(img):
    if img.ndim == 2:
        return img[np.newaxis, :, :]
    else:
        return img.transpose(2,0,1)

def load_image_gray(path, size=None):
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    return to_chw(x)

def save_image(arr, path):
    from PIL import Image
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr.transpose(1,2,0)
    img = Image.fromarray(np.clip(arr*255,0,255).astype("uint8"))
    img.save(path)
