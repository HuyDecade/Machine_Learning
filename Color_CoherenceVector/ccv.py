import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_ccv(image, tau=100):
    """
    Tính toán Color Coherence Vector (CCV).
    image: Ảnh đầu vào (ảnh màu).
    tau: Ngưỡng để phân biệt giữa pixel kết dính và không kết dính.
    """
    # Chuyển ảnh từ RGB sang HSV để phân tích màu tốt hơn
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # Tạo histogram màu (ở đây chỉ dùng kênh H)
    hist, bins = np.histogram(h.ravel(), bins=16, range=[0, 180])
    
    # Đặt ngưỡng vùng kết dính
    coherent = np.zeros_like(hist)   # Vector coherent (pixel kết dính)
    incoherent = np.zeros_like(hist) # Vector incoherent (pixel không kết dính)
    
    # Duyệt qua từng pixel và tính kích thước vùng của nó
    for i in range(16):  # Với mỗi bin trong histogram
        # Tạo mặt nạ cho từng vùng màu cụ thể và chuyển đổi sang kiểu `np.uint8`
        mask = ((h >= i * 180 // 16) & (h < (i + 1) * 180 // 16)).astype(np.uint8)
        
        # Tìm các vùng liên kết (connected components)
        num_labels, labeled = cv2.connectedComponents(mask)
        
        # Duyệt qua từng nhãn vùng để xác định coherent và incoherent
        for label in range(1, num_labels):  # Bỏ qua nhãn 0 vì là nền
            region_size = np.sum(labeled == label)
            if region_size >= tau:  # Nếu kích thước >= ngưỡng tau -> kết dính
                coherent[i] += region_size
            else:  # Nếu không -> không kết dính
                incoherent[i] += region_size

    return coherent, incoherent

# Đường dẫn đến ảnh (Chỉnh sửa đường dẫn để tránh lỗi escape)
image_path = r'd:\Machine_Leanring\pexels-fotofyn-634613.jpg'
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh được đọc thành công
if image is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")

# Tính toán CCV
coherent_vector, incoherent_vector = compute_ccv(image)

# In kết quả
print("Coherent Vector: ", coherent_vector)
print("Incoherent Vector: ", incoherent_vector)

# Hiển thị ảnh
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Sample Image")
plt.show()
