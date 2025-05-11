import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import load_model
from utils.select_image_file import select_image_file
import threading, queue
import cv2
import mediapipe as mp
import numpy as np


# Load model
gender_model = load_model("model/gender_model.keras")

# Tạo queue
gender_queue = queue.Queue(maxsize=1)
gender_result_queue = queue.Queue(maxsize=1)

# Khởi tạo Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)


def predict_gender(face_img):
    try: 
        if face_img.size == 0:
            return "Unknown"
        resized = cv2.resize(face_img, (64, 64)) / 255.0
        input_img = resized.reshape(1, 64, 64, 3)
        gender_pred = gender_model.predict(input_img, verbose=0)
        probability = gender_pred[0][0]
        if probability > 0.5:
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



threading.Thread(target=gender_predict_thread, daemon=True).start()


img_path = select_image_file()
if not img_path:
    print("Không chọn ảnh nào, thoát chương trình.")
    exit(0)

image = cv2.imread(img_path)
if image is None:
    print("Không đọc được ảnh.")
    exit(0)

ih, iw, _ = image.shape
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = face_mesh.process(rgb)

if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:

        # Tính toán vùng khuôn mặt
        x_coords = [int(pt.x * iw) for pt in face_landmarks.landmark]
        y_coords = [int(pt.y * ih) for pt in face_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        face_crop = image[y_min:y_max, x_min:x_max]

        if face_crop.size == 0:
            continue

        # Gửi ảnh vào queue và chờ kết quả
        gender_queue.put(face_crop)
        gender = gender_result_queue.get()

         # Hiển thị kết quả
        cv2.putText(image, f"Gender: {gender}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

 

# Gửi tín hiệu kết thúc và chờ worker thread dừng
gender_queue.put(None)
gender_result_queue.get()  # Đợi xác nhận dừng từ worker

# Hiển thị ảnh kết quả
cv2.imshow("Gender Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
