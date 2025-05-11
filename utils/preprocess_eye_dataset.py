import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle

def preprocess_eye_data(dataset_path='dataset/driver-drowsiness-detection'):
    data = []
    labels = []

    for label, category in enumerate(['closed_eye', 'open_eye']):
        folder = os.path.join(dataset_path, category)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (24, 24))
            img = img / 255.0  # Normalize
            data.append(img)
            labels.append(label)

    X = np.array(data).reshape(-1, 24, 24, 1)
    y = to_categorical(labels)

    # Lưu lại dữ liệu đã xử lý để dùng cho file train
    with open('processed_eye_data.pkl', 'wb') as f:
        pickle.dump((X, y), f)

    print("Data preprocessing completed and saved to processed_eye_data.pkl")

if __name__ == "__main__":
    preprocess_eye_data()
