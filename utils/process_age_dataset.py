import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

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
            continue

        # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize ảnh về kích thước 64x64
        img = cv2.resize(img, (64, 64))

        images.append(img)
        labels.append(gender)


    # Chuyển danh sách thành mảng numpy và chuẩn hóa giá trị pixel về [0,1]
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels)

    # Chia thành tập train và test (ví dụ 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print('Số lượng ảnh train:', X_train.shape[0])
    print('Số lượng ảnh test:', X_test.shape[0])

    return X_train, X_test, y_train, y_test