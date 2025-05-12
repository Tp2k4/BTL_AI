import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from utils.crop_face import crop_face
import matplotlib.pyplot as plt 


def process_age_dataset():
    data_dir = "dataset/gender"  

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

        # Lấy ảnh
        img_path = os.path.join(data_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("lỗi")
            continue
        else: 
            print("thành công")
        
        #Cắt mặt
        result = crop_face(img)
        if result is not None:
            cropped_face = result[0]

        if cropped_face is None:
            print("lỗi")
            continue
        else: 
            print("thành công")

        # Chuyển ảnh sang grayscale
        img_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

        # Thêm chiều cho ảnh để có dạng (64, 64, 1)
        img_gray = np.expand_dims(img_gray, axis=-1)

        # thêm vào tập X_train và y_train
        images.append(img_gray)
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



