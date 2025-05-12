import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from utils.crop_face import crop_face

def process_gender_dataset():

    data_dir = 'dataset/gender' 

    # Danh sách để chứa ảnh và nhãn
    images = []
    labels = []

    # Duyệt qua tất cả file ảnh trong thư mục
    for filename in os.listdir(data_dir):
        if not filename.endswith('.jpg'):
            continue

        # Tách nhãn giới tính từ tên file
        parts = filename.split('_')
        if len(parts) < 2:
            continue
        gender = int(parts[1]) # 0 = nam, 1 = nữ

        # Đọc ảnh màu với OpenCV
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

        images.append(img_gray)
        labels.append(gender)


    # Chuyển danh sách thành mảng numpy và chuẩn hóa giá trị pixel về [0,1]
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels)

    # Chia thành tập train và test (ví dụ 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print('Số lượng ảnh train:', X_train.shape[0])
    print('Số lượng ảnh test:', X_test.shape[0])

    return X_train, X_test, y_train, y_test


