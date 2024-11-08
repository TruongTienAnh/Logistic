import numpy as np
import pandas as pd
from scipy.optimize import minimize


# Bước 1: Chuẩn bị dữ liệu
# Đọc dữ liệu từ tệp Excel
file_path = 'Logistic1.xlsx'  # Đường dẫn đến tệp Excel
sheet_name = 'LR'  # Tên sheet bạn muốn đọc (thay đổi nếu cần)

# Sử dụng pandas để đọc tệp Excel
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Chuyển đổi DataFrame thành NumPy array
data= df.to_numpy()

initial_weights = np.zeros(data.shape[1])  # Khởi tạo trọng số với số 0 (bao gồm B0)

# Bước 2: Hàm sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Giới hạn để tránh overflow
    return 1 / (1 + np.exp(-z))

# Bước 3: Tính xác suất dự đoán
def calculate_probability(data, weights):
    probabilities = []
    for row in data:
        z = weights[0] + np.dot(row[:-1], weights[1:])  # B0 + B1*X1 + B2*X2 + ... + B13*X13
        prob = sigmoid(z)  # Chuyển đổi sang xác suất
        probabilities.append(prob)
    return probabilities

# Bước 4: Tính hàm hợp lý
def log_likelihood(weights, data):
    probabilities = calculate_probability(data, weights)
    total_cost = 0
    for i, row in enumerate(data):
        y = row[-1]  # Nhãn thực tế
        p = probabilities[i]  # Xác suất dự đoán
        cost = y * np.log(p) + (1 - y) * np.log(1 - p)
        total_cost += cost
    return -total_cost  # Trả về giá trị âm để tối thiểu hóa

# Bước 5: Tối ưu hóa hàm hợp lý
def solve(data, initial_weights):
    result = minimize(log_likelihood, initial_weights, args=(data,), method='BFGS')  # Sử dụng BFGS để tối ưu hóa

    return result.x  # Trả về trọng số tối ưu


# Bước 6: Huấn luyện mô hình và dự đoán
trained_weights = solve(data, initial_weights)

# Dự đoán trên tập kiểm tra
def predict(data, weights):
    probabilities = calculate_probability(data, weights)
    return (np.array(probabilities) >= 0.5).astype(int), probabilities  # Trả về cả dự đoán và xác suất

# Dự đoán trên toàn bộ dữ liệu
predictions, probabilities = predict(data, trained_weights)

# In ra xác suất dự đoán cho từng mẫu
print("Trọng số tối ưu:")
for i, weight in enumerate(trained_weights):
    print(f"B{i}: {weight:.4f}")  # In với độ chính xác 4 chữ số thập phân

print("\nXác suất dự đoán cho từng mẫu:")
for i, prob in enumerate(probabilities):
    print(f"Mẫu {i + 1}: Xác suất = {prob:.4f}, Dự đoán = {predictions[i]}")

# Tính độ chính xác
accuracy_manual = np.mean(predictions == data[:, -1])
print("\nĐộ chính xác (Code tay):", accuracy_manual)

file_path = 'Logistic1.xlsx'  # Đường dẫn đến tệp Excel
sheet_name = 'Sheet1'  # Tên sheet bạn muốn đọc (thay đổi nếu cần)

# Sử dụng pandas để đọc tệp Excel
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Chuyển đổi DataFrame thành NumPy array
dataTest= df.to_numpy()
predictionss, probabilitiess = predict(dataTest, trained_weights)
print("\nXác suất dự đoán cho từng mẫu test:")
for i, prob in enumerate(probabilitiess):
    print(f"Mẫu {i + 1}: Xác suất = {prob:.4f}, Dự đoán = {predictionss[i]}")