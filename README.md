# 🧠 Công Cụ Xử Lý Ảnh — Nhận Dạng Chữ Số & Hình Học

Một công cụ xử lý ảnh số có khả năng nhận dạng chữ số viết tay và phát hiện hình dạng hình học, được xây dựng hoàn toàn từ đầu bằng cách sử dụng CNN dựa trên NumPy (không sử dụng TensorFlow/PyTorch).

**👤 Tác giả:** [translate:Nguyễn Tất Thắng] – B23DCKH107  
**📚 Môn học:** Xử lý ảnh số – INT13146  
**🎓 Trường:** Học viện Công nghệ Bưu Chính Viễn Thông

---

## 📖 Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Tính Năng](#tính-năng)
- [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
- [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
- [Cài Đặt](#cài-đặt)
- [Bắt Đầu Nhanh](#bắt-đầu-nhanh)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)

---

## 🎯 Giới Thiệu

Dự án này xây dựng một hệ thống xử lý ảnh số toàn diện với hai mô-đun nhận dạng chính:

1. **Nhận dạng chữ số viết tay (MNIST)** – Mô hình CNN được huấn luyện trên bộ dữ liệu MNIST để phân loại chữ số (0–9)
2. **Phát hiện hình dạng hình học** – CNN tùy chỉnh để nhận dạng các hình dạng cơ bản (hình chữ nhật, hình tròn) từ ảnh tải lên

Tất cả các thành phần mạng nơ-ron được triển khai **từ đầu bằng NumPy**, bao gồm lan truyền tiến/lùi, tối ưu hóa và các hàm kích hoạt. Giao diện đồ họa tương tác được xây dựng bằng Tkinter cung cấp dự đoán thời gian thực với điểm độ tin cậy.

---

## ✨ Tính Năng

### 🔢 Chế Độ Nhận Dạng Chữ Số
- **Vẽ Tương Tác Trên Canvas** – Vẽ các chữ số tự do trên canvas 280×280 pixel
- **Tiền Xử Lý Kiểu MNIST** – Chuẩn hóa ảnh tự động:
  - Cắt vùng nội dung
  - Phóng to/thu nhỏ chiều lớn nhất thành 20px
  - Căn giữa trong khung 28×28
- **Dự Đoán Thời Gian Thực** – Phân loại chữ số (0–9) tức thời với phần trăm độ tin cậy
- **Xóa & Đặt Lại** – Điều khiển thân thiện với người dùng để quản lý canvas

### 🔷 Chế Độ Nhận Dạng Hình Dạng
- **Tải Ảnh Lên** – Tải ảnh từ đĩa
- **Hiển Thị Xem Trước** – Xác nhận trực quan trước khi xử lý
- **Thay Đổi Kích Thước 32×32** – Tiền xử lý đầu vào được chuẩn hóa
- **Phân Loại Nhị Phân** – Phát hiện Hình chữ nhật vs. Hình tròn
- **Điểm Độ Tin Cậy** – Chỉ báo chắc chắn dự đoán

### 🎨 Giao Diện Người Dùng
- **GUI Tkinter Hiện Đại** – Giao diện sạch sẽ, đáp ứng
- **Chuyển Đổi Hai Chế Độ** – Chuyển đổi liền mạch giữa các chế độ nhận dạng chữ số và hình dạng
- **Chỉ Báo Trạng Thái** – Phản hồi xử lý thời gian thực
- **Xử Lý Lỗi** – Thông báo lỗi thân thiện với người dùng

---

## 🛠️ Công Nghệ Sử Dụng

### Framework Cốt Lõi
- **NumPy** – Triển khai mạng nơ-ron hoàn chỉnh từ đầu
- **Pillow (PIL)** – Xử lý và thao tác ảnh
- **Tkinter** – Framework GUI (bao gồm sẵn với Python)

### Các Thành Phần Mạng Nơ-Ron (Pure NumPy)
- Lớp Tích Chập (Conv2D)
- Lớp Max Pooling (MaxPool2D)
- Hàm Kích Hoạt (ReLU, Softmax)
- Lớp Kết Nối Đầy Đủ (Dense)
- Hàm Mất Mát (Softmax Cross-Entropy)
- Tối Ưu Hóa (SGD với động lượng tùy chọn)
- Tuần tự hóa Mô Hình (lưu/tải trọng số)

### Không Sử Dụng Thư Viện ML Bên Ngoài
❌ TensorFlow  
❌ PyTorch  
❌ Keras  
✅ Pure NumPy + Python Tiêu Chuẩn

---

## 🏗️ Kiến Trúc Mô Hình

### CNN Chữ Số (Nhận Dạng MNIST)

```
Đầu vào: Ảnh xám 28×28
    ↓
Conv2D(8 filters, kernel=3×3, padding=1) → ReLU
    ↓
MaxPool2D(2×2)
    ↓
Conv2D(16 filters, kernel=3×3, padding=1) → ReLU
    ↓
MaxPool2D(2×2)
    ↓
Flatten → Dense(784 → 64) → ReLU
    ↓
Dense(64 → 10) → Softmax
    ↓
Đầu ra: Logits cho các chữ số 0–9
```

**Hình Dạng Đầu Ra:** (10,)

---

### CNN Hình Dạng (Phát Hiện Hình Chữ Nhật/Tròn)

```
Đầu vào: Ảnh xám 32×32
    ↓
Conv2D(8 filters, kernel=3×3) → ReLU
    ↓
MaxPool2D(2×2)
    ↓
Conv2D(16 filters, kernel=3×3) → ReLU
    ↓
MaxPool2D(2×2)
    ↓
Flatten → Dense(1024 → 32) → ReLU
    ↓
Dense(32 → 2) → Softmax
    ↓
Đầu ra: [Xác suất Hình chữ nhật, Xác suất Hình tròn]
```

**Hình Dạng Đầu Ra:** (2,)

---

## 📦 Cài Đặt

### Yêu Cầu Hệ Thống
- **Python:** 3.8 trở lên
- **Hệ điều hành:** Windows, macOS, hoặc Linux
- **RAM:** Tối thiểu 2GB (4GB+ được khuyên dùng)
- **Không Gian Đĩa:** ~100MB cho các bộ dữ liệu và checkpoint

### Bước 1: Clone Kho Lưu Trữ

### Bước 2: Tạo Môi Trường Ảo (Được Khuyến Khích)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài Đặt Các Thư Viện Phụ Thuộc

```bash
pip install -r requirements.txt
```

### Bước 4: Xác Minh Cài Đặt

```bash
python -c "import numpy, PIL, scipy; print('✓ Tất cả các phụ thuộc đã được cài đặt')"
```

---

## 🚀 Bắt Đầu Nhanh

### Tùy Chọn 1: Sử Dụng Các Mô Hình Được Đào Tạo Trước (Được Khuyên Dùng Cho Người Dùng Lần Đầu)

```bash
# Chỉ cần chạy GUI với các checkpoint hiện có
python gui.py --ckpt_digit checkpoints/mnist_best.npz \
              --ckpt_shape checkpoints/shapes_best.npz
```

### Tùy Chọn 2: Huấn Luyện Mô Hình Từ Đầu

```bash
# Huấn luyện mô hình nhận dạng chữ số MNIST
python train.py --task mnist --epochs 10 --batch 64 --lr 0.001

# Huấn luyện mô hình nhận dạng hình dạng
python train.py --task shapes --epochs 20 --batch 64 --lr 0.001

# Khởi chạy GUI với các mô hình được đào tạo mới
python gui.py --ckpt_digit checkpoints/mnist_best.npz \
              --ckpt_shape checkpoints/shapes_best.npz
```

---

## 📖 Hướng Dẫn Sử Dụng

### Chạy Ứng Dụng

```bash
python gui.py --ckpt_digit checkpoints/mnist_best.npz \
              --ckpt_shape checkpoints/shapes_best.npz
```

### Chế Độ 1: Nhận Dạng Chữ Số Viết Tay

1. Chọn tab **"Digit Draw"**
2. Vẽ một chữ số (0–9) trên canvas
3. Nhấp nút **"Predict"**
4. Xem kết quả dự đoán và điểm độ tin cậy
5. Nhấp **"Clear"** để đặt lại canvas cho chữ số tiếp theo

### Chế Độ 2: Nhận Dạng Hình Dạng Hình Học

1. Chọn tab **"Shape Upload"**
2. Nhấp **"Upload Image"** để chọn file ảnh
3. Bản xem trước hiển thị ảnh được tải lên
4. Nhấp **"Predict"** để phân loại hình dạng
5. Xem kết quả phân loại (Hình chữ nhật hoặc Hình tròn)

**Định Dạng Ảnh Được Hỗ Trợ:**
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.gif`

---

## 📂 Cấu Trúc Dự Án

```
CNN-BTL-xla/
│
├── 📁 core/
│   ├── layers.py              # Lớp Conv2D, MaxPool2D, Dense, Flatten
│   ├── losses.py              # Hàm mất mát Softmax Cross-Entropy
│   ├── model.py               # Lớp mô hình Sequential + forward/backward
│   ├── optim.py               # Bộ tối ưu hóa SGD với động lượng
│   └── utils.py               # Tiện ích tính toán tại các lớp mạng CNN
│
├── 📁 datasets               # Dữ liệu để huấn luyện và kiểm thử
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
│   ├── train-images.idx3-ubyte
│   └── train-images.idx1-ubyte
│
├── 📁 datasets/
│   ├── mnist.py               # Trình tải bộ dữ liệu MNIST (60K train, 10K test)
│   ├── shapes.py              # Bộ sinh dữ liệu tổng hợp (hình chữ nhật/tròn)
│   └── utils.py               # Tiện ích tiền xử lý ảnh
│
├── 📁 checkpoints/
│   ├── mnist_best.npz         # Trọng số mô hình MNIST được đào tạo trước
│   └── shapes_best.npz        # Trọng số mô hình hình dạng được đào tạo trước
│
├── 📁 report/                 # Báo cáo bài tập lớn
│
├── train.py                   # Script huấn luyện mô hình
├── gui.py                     # Ứng dụng GUI Tkinter
├── infer.py                   # Chương trình dự đoán (không GUI)
├── requirements.txt           # Các phụ thuộc Python
└── README.md
```

### Chi Tiết Mô-đun Cốt Lõi

#### `core/layers.py`
Triển khai các lớp mạng nơ-ron cơ bản từ đầu:
- `Conv2D` – Tích chập 2D với hỗ trợ đệm
- `MaxPool2D` – Hoạt động max pooling
- `Dense` – Lớp được kết nối đầy đủ
- `Flatten` – Làm phẳng thành 1D
- `ReLU` – Kích hoạt Rectified Linear Unit
- `Softmax` – Kích hoạt Softmax cho phân loại

#### `core/model.py`
Lớp mô hình `Sequential` hỗ trợ:
- Lan truyền tiến
- Lan truyền lùi với tính toán gradient
- Xử lý theo lô
- Tuần tự hóa trọng số (lưu/tải định dạng .npz)

#### `datasets/shapes.py`
Tạo bộ dữ liệu tổng hợp:
- Vẽ các hình chữ nhật hoàn hảo với biến thể (quay, kích thước, vị trí)
- Vẽ các hình tròn hoàn hảo với biến thể (bán kính, vị trí, nhiễu)
- Xuất ảnh xám 32×32 với nhãn nhị phân
- Tạo bộ huấn luyện/xác thực/kiểm tra cân bằng

---

## 📧 Liên Hệ & Hỗ Trợ

Để đặt câu hỏi, báo cáo lỗi hoặc đề xuất:

- **Tác giả:** Nguyễn Tất Thắng
- **Email:** ThangNT.B23KH107@stu.ptit.edu.vn
- **Trường:** Học viện Công nghệ Bưu Chính Viễn Thông

---

*Cập Nhật Lần Cuối: Tháng 12 năm 2025*  
*Được Tạo Bằng Bởi Nguyễn Tất Thắng*
