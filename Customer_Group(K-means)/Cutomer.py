# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Đường dẫn tới tập dữ liệu cục bộ
data_path = 'D:/CODE/BT01/Mall_Customers.csv'

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv(data_path)

# Hiển thị 5 dòng đầu tiên của tập dữ liệu
print("Dữ liệu ban đầu:")
print(data.head())

# Lấy hai cột cần sử dụng cho phân cụm: Annual Income và Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Xác định số cụm (clusters) mong muốn bằng thuật toán K-Means
kmeans = KMeans(n_clusters=5, random_state=42)  # 5 cụm
kmeans.fit(X)

# Gán nhãn cụm cho từng khách hàng
data['Cluster'] = kmeans.labels_

# Hiển thị kết quả phân cụm
print("\nKết quả phân cụm với 5 nhóm:")
print(data.head())

# Trực quan hóa các cụm
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
