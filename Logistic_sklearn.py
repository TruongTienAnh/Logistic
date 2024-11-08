# Nhập các thư viện cần thiết
from sklearn.model_selection import train_test_split  # Thư viện chia dữ liệu
from sklearn.linear_model import LogisticRegression   # Thư viện hồi quy logistic
from sklearn.metrics import accuracy_score, classification_report  # Thư viện đánh giá hiệu suất mô hình
from sklearn.preprocessing import StandardScaler  # Thư viện chuẩn hóa dữ liệu
import pandas as pd  # Thư viện xử lý dữ liệu
import numpy as np

# Bước 1: Chuẩn bị dữ liệu
df = pd.read_excel('Logistic1.xlsx', sheet_name='LR')  # Đọc dữ liệu từ file Excel
print("Số lượng mẫu trong tập dữ liệu:", df.shape[0])  # In ra số lượng mẫu trong tập dữ liệu

# Tách dữ liệu thành các cột đặc trưng (X) và cột nhãn mục tiêu (y)
X = df.iloc[:, :-1]  # Các cột đặc trưng (tất cả các cột ngoại trừ cột cuối cùng)
y = df.iloc[:, -1]   # Cột nhãn mục tiêu (cột cuối cùng)

# Bước 2: Chuẩn hóa dữ liệu
scaler = StandardScaler()  # Khởi tạo đối tượng chuẩn hóa
X_scaled = scaler.fit_transform(X)  # Chuẩn hóa dữ liệu bằng cách tính trung bình và độ lệch chuẩn

# Bước 3: Chia dữ liệu đã chuẩn hóa thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# test_size=0.2 có nghĩa là 20% dữ liệu sẽ được sử dụng cho kiểm tra và 80% cho huấn luyện
# random_state=42 đảm bảo rằng kết quả chia dữ liệu là nhất quán giữa các lần chạy

# Bước 4: Khởi tạo và huấn luyện mô hình hồi quy logistic
model = LogisticRegression(solver='liblinear', max_iter=2000)  # Tăng max_iter lên 2000 để tránh cảnh báo hội tụ
model.fit(X_train, y_train)  # Huấn luyện mô hình với dữ liệu huấn luyện

# Bước 5: Dự đoán trên tập kiểm tra
predictions = model.predict(X_test)  # Dự đoán nhãn cho tập kiểm tra

# Bước 6: Tính độ chính xác
accuracy = accuracy_score(y_test, predictions)  # Tính độ chính xác của mô hình
print("Độ chính xác:", accuracy)  # In ra độ chính xác

# Bước 7: In báo cáo phân loại chi tiết
print("\nBáo cáo phân loại:\n", classification_report(y_test, predictions))
# In ra các chỉ số như precision, recall, f1-score cho từng lớp trong dữ liệu

# Bước 8: In ra trọng số của mô hình
print("Trọng số của mô hình (B1, B2, ...):", model.coef_)  # In ra các trọng số cho mỗi đặc trưng
print("Trọng số bias (B0):", model.intercept_)  # In ra trọng số bias

# Bước 9: Tính xác suất dự đoán
probabilities = model.predict_proba(X_test)[:, 1]  # Lấy xác suất cho lớp dương (cột thứ hai)
print("\nXác suất dự đoán cho từng mẫu:")  # In ra tiêu đề cho xác suất dự đoán
for i, prob in enumerate(probabilities):  # Duyệt qua từng xác suất
    print(f"Mẫu {i + 1}: Xác suất = {prob:.4f}, Dự đoán = {predictions[i]}")
    # In ra xác suất dự đoán cho từng mẫu và nhãn dự đoán tương ứng

