
# [DS200] Lab 04: Truyền ảnh CIFAR-10 và huấn luyện mô hình học máy với Spark Streaming

## Mục tiêu
Xây dựng hệ thống truyền dữ liệu ảnh thời gian thực từ một máy chủ giả lập đến Spark Streaming. Sau đó, thực hiện tiền xử lý và huấn luyện mô hình học máy như **SVM** hoặc **KMeans** trên các batch dữ liệu.

---

## Cấu trúc thư mục
```
project/
├── models/
│   ├── __init__.py
│   ├── kmeansClustering.py
│   └── svm.py
├── transforms/
│   ├── __init__.py
│   ├── color_shift.py
│   ├── normalize.py
│   ├── random_flips.py
│   ├── resize.py
│   └── transforms.py
├── dataloader.py
├── main.py
├── stream.py
├── trainer.py
├── cifar-10-python.tar.gz       # File dữ liệu gốc
└── cifar-10-batches-py/         # Đã giải nén
```

---

## 1. Yêu cầu hệ thống

### Phần mềm

- Python ≥ 3.8
- Java ≥ 17
- Apache Spark ≥ 3.4.1 hoặc Spark 4.0.0
- Hệ điều hành: Windows/Linux/macOS

### Cài đặt thư viện Python

```bash
pip install numpy scikit-learn pyspark tqdm joblibspark matplotlib torch
```

---

## 2. Dữ liệu CIFAR-10

### Tải

- [Tải trực tiếp tại đây](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Hoặc dùng lệnh:

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

### Giải nén

Giải nén thành thư mục `cifar-10-batches-py/` chứa:
```
├── data_batch_1
├── data_batch_2
├── ...
├── test_batch
```

---

## 3. Mô tả hoạt động hệ thống

### `stream.py` – Gửi dữ liệu từ server giả lập

- Đọc file `.pkl` từ CIFAR-10
- Đóng gói từng batch ảnh thành JSON
- Gửi qua socket TCP đến Spark

### `main.py` – Nhận dữ liệu và huấn luyện mô hình

- Nhận ảnh từ stream qua Spark Streaming
- Tiền xử lý ảnh (`resize`, `normalize`, `flip`,...)
- Chuyển về vector → DataFrame
- Huấn luyện mô hình (SVM hoặc KMeans)

---

## 4. Cách chạy chương trình

### Terminal 1: Chạy Spark + huấn luyện

```bash
python main.py 

### Terminal 2: Chạy server gửi dữ liệu

```bash
python stream.py     --folder cifar-10-batches-py     --batch-size 100     --split train     --sleep 1
```

Giải thích tham số:
- `--folder`: thư mục chứa ảnh
- `--batch-size`: số ảnh mỗi lần gửi
- `--split`: `train` hoặc `test`
- `--sleep`: thời gian nghỉ (giây) giữa các batch

---

## 5. Quá trình xử lý dữ liệu

### `stream.py`

- Đọc dữ liệu từ file `data_batch_x`
- Gửi từng batch qua socket

### `dataloader.py`

- Nhận dòng JSON
- Tiền xử lý ảnh: `resize`, `normalize`, `flip`
- Chuyển ảnh thành `DenseVector` để dùng trong MLlib

### `trainer.py`

- Tạo Spark Streaming Context
- Nhận ảnh → DataFrame
- Gọi `.train()` hoặc `.predict()` của mô hình tương ứng

---

## 6. Kết quả đầu ra

### Khi huấn luyện (`--mode train`)
```text
==========
Ví dụ với mô hình SVM:
Predictions = [7 9 3 4 5 0 5 2 3 2 6 0 4 9 0 7 0 9 7 2 6 4 6 9 5 4 7 0 6 8 8 9 9 9 0 9 8
 6 4 8 1 9 1 0 5 8 6 9 6 0 8 1 3 9 4 8 4 3 2 6 0 8 9 9 4 3 0 2 4 4 0 3 5 7
 5 7 7 9 0 9 5 3 8 2 4 2 3 1 2 8 9 2 8 1 4 2 0 4 5 4]
Accuracy = 1.0
Precision = 1.0
Recall = 1.0
F1 Score = 1.0
==========
Total Batch Size of RDD Received: 100
++++++++++++++++++++
```

### Khi dự đoán (`--mode predict`)
- In kết quả từng batch
- Tự động tính **trung bình các chỉ số**
- **Lưu kết quả tổng hợp ra file `predict_summary.csv`**

```csv
Ví dụ với mô hình SVM:
Accuracy,Precision,Recall,F1_Score
1.0,1.0,1.0,1.0
```

---
## 7. Mở rộng

- Thay TCP bằng Kafka
- Dùng CNN (PyTorch) thay vì SVM
- Lưu mô hình sau khi train
- Giao diện real-time bằng Flask hoặc Streamlit

---

## Tác giả

Bài Lab được thực hiện bởi Hồ Ngọc Mai, MSSV: 22520839, Môn học: Phân tích Dữ liệu lớn, GVHD: Nguyễn Hiếu Nghĩa

Mục tiêu: Hiểu rõ pipeline thời gian thực trong học máy, tích hợp Spark Streaming & MLlib.

---

## Tài liệu tham khảo

- [PySpark Streaming Guide](https://spark.apache.org/docs/latest/streaming-programming-guide.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
