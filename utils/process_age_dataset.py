import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def process_age_dataset():
    data_dir = 'dataset/gender'  

    images = []
    ages = []

    for filename in os.listdir(data_dir):
        if not filename.endswith('.jpg'):
            continue

        # Phân tích tên file để lấy thông tin tuổi
        parts = filename.split('_')
        if len(parts) < 4:  # Kiểm tra định dạng tên file
            continue

        try:
            age = int(parts[0])  # Tuổi là phần đầu tiên trong tên file
            if age < 0 or age > 116:  # Kiểm tra khoảng tuổi hợp lệ
                continue
        except ValueError:
            continue

        # Xử lý ảnh
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))

        images.append(img)
        ages.append(age)

    # Chuẩn hóa dữ liệu
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(ages)

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=None  # Không phân tầng do đây là bài toán hồi quy
    )
    
    print(f'Số lượng ảnh huấn luyện: {X_train.shape[0]}')
    print(f'Số lượng ảnh kiểm tra: {X_test.shape[0]}')
    print(f'Phân bố tuổi - Min:{y.min()}, Max:{y.max()}, Mean:{y.mean():.1f}')

    return X_train, X_test, y_train, y_test
