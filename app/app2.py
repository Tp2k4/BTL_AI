import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import load_model
from utils.select_image_file import select_image_file
from utils.crop_face import crop_face
import threading, queue
import cv2
import mediapipe as mp
import numpy as np


# Load model
gender_model = load_model("model/gender_model.h5")
age_model = load_model("model/age_model.h5")

# Tạo queue
gender_queue = queue.Queue(maxsize=1)
gender_result_queue = queue.Queue(maxsize=1)
age_queue = queue.Queue(maxsize=1)
age_result_queue = queue.Queue(maxsize=1)

# Khởi tạo Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)


def predict_gender(face_img):
    try: 
        if face_img.size == 0:
            return "Unknown"
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)) / 255.0
        input_img = resized.reshape(1, 64, 64, 1)
        gender_pred = gender_model.predict(input_img, verbose=0)

        probability = gender_pred[0][0]
        print(f"Probability Female: {probability*100:.2f}%")
        print(f"Probability Male: {(1 - probability)*100:.2f}%")
        if probability < 0.5:
            gender = "Male"
        else: 
            gender = "Female"
        return gender
    
    except Exception as e:
        print(f"Lỗi dự đoán giới tính: {e}")
        return "Unknown"


def gender_predict_thread():
    while True:
        face_img = gender_queue.get()
        if face_img is None:
            break
        try:
            gender = predict_gender(face_img)
            gender_result_queue.put(gender)
        except:
            gender_result_queue.put("Unknown")

    gender_result_queue.put(None)



def predict_age(face_img):
    try:
        if face_img.size == 0:
            return "Unknown"
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)) / 255.0
        input_img = resized.reshape(1, 64, 64, 1)
        age_pred = age_model.predict(input_img, verbose=0)

        predicted_age = age_pred[0][0]
        predicted_age = predicted_age * 116
        # In 30 xác suất cao nhất
        margin = 5  # Sai số dự kiến, bạn có thể điều chỉnh dựa trên MAE thực tế

        age_min = max(0, int(predicted_age - margin))
        age_max = int(predicted_age + margin)

        print(f"Predicted age range: {age_min} - {age_max}")
        return f"{age_min} - {age_max}"
    
    except Exception as e:
        print(f"Lỗi dự đoán tuổi: {e}")
        return "Unknown"



def age_predict_thread():
    while True:
        face_img = age_queue.get()
        if face_img is None:
            break
        try:
            age = predict_age(face_img)
            age_result_queue.put(age)
        except Exception as e:
            print(f"Lỗi trong thread dự đoán tuổi: {e}")
            age_result_queue.put("Unknown")

    age_result_queue.put(None)


threading.Thread(target=gender_predict_thread, daemon=True).start()
threading.Thread(target=age_predict_thread, daemon=True).start()


img_path = select_image_file()
if not img_path:
    print("Không chọn ảnh nào, thoát chương trình.")
    exit(0)

image = cv2.imread(img_path)
if image is None:
    print("Không đọc được ảnh.")
    exit(0)

result = crop_face(image)
if result is not None:
    face_crop, x_min, x_max, y_min, y_max = result
    
    # Gửi ảnh vào 2 queue riêng biệt
    gender_queue.put(face_crop)
    age_queue.put(face_crop)

    # Lấy kết quả dự đoán từ 2 queue kết quả
    gender = gender_result_queue.get()
    age = age_result_queue.get()


    # Hiển thị kết quả
    label = f"Gender: {gender}, Age: {age}"
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
else:
    print("Lỗi không phát hiện được khuôn mặt")
 

# Gửi tín hiệu kết thúc và chờ worker thread dừng
gender_queue.put(None)
gender_result_queue.get()  

age_queue.put(None)
age_result_queue.get()

# Hiển thị ảnh kết quả
cv2.imshow("Gender Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
