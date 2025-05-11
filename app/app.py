# [1. Khởi tạo: Load model, hàng đợi, Mediapipe và biến trạng thái] 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading, queue, time
from tensorflow.keras.models import load_model
from collections import deque
from utils.select_image_file import select_image_file

# Load models
eye_model = load_model("model/eye_state_model.h5")
emotion_model = load_model("model/emotion_model.keras")
age_model = load_model("model/eye_state_model.h5", compile=False)
gender_model = load_model("model/gender_model.keras")

# Queues
eye_queue = queue.Queue(maxsize=2)
eye_result_queue = queue.Queue(maxsize=2)
emotion_queue = queue.Queue(maxsize=1)
emotion_result_queue = queue.Queue(maxsize=1)
speech_queue = queue.Queue()
gender_queue = queue.Queue(maxsize=1)
gender_result_queue = queue.Queue(maxsize=1)

# Mediapipe
# Trả về một module có chứa các lớp và hàm liên quan đến việc xử lý khuôn mặt
mp_face = mp.solutions.face_mesh
# trả về một đối tượng có thể được sử dụng để nhận diện và phân tích các đặc điểm khuôn mặt
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# Trạng thái
prev_label = None
label_history = deque(maxlen=10)  # Tăng chiều dài để mượt hơn
current_eye_state = "Opening"
current_emotion = "Neutral"
current_age = None
current_gender = None

last_eye_state_change = time.time()
eye_state_interval = 1.0  # giây


# [2. Luồng phát âm bằng pyttsx3] 
def speech_loop():
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# [3. Luồng dự đoán trạng thái mắt] 
def eye_predict_thread():
    while True:
        eye_pair = eye_queue.get()
        if eye_pair is None:
            break
        left_eye, right_eye = eye_pair

        def predict_eye(eye_img):
            if eye_img.size == 0:
                return 1
            gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (24, 24)) / 255.0
            input_img = resized.reshape(1, 24, 24, 1)
            pred = eye_model.predict(input_img, verbose=0)
            return np.argmax(pred)

        try:
            left = predict_eye(left_eye)
            right = predict_eye(right_eye)
            label = 0 if left == 0 or right == 0 else 1
            eye_result_queue.put(label)
        except:
            eye_result_queue.put(1)

# [4. Xử lý kết quả mắt: thay đổi trạng thái + phát âm] 
def handle_eye_state():
    global prev_label, current_eye_state, last_eye_state_change
    if not eye_result_queue.empty():
        label = eye_result_queue.get()
        label_history.append(label)
        smoothed = 1 if label_history.count(1) > label_history.count(0) else 0

        now = time.time()
        if smoothed != prev_label and (now - last_eye_state_change) > eye_state_interval:
            prev_label = smoothed
            last_eye_state_change = now
            if smoothed == 0:
                current_eye_state = "Closing"
                speak("closing")
            else:
                current_eye_state = "Opening"
                speak("opening")

# [5. Hàm phát âm: kiểm tra trùng lặp để tránh lặp lại âm thanh] 
def speak(text):
    if speech_queue.empty():
        speech_queue.put(text)
    else:
        current_speech = speech_queue.queue[-1]
        if current_speech != text:
            speech_queue.put(text)

# [6. Luồng dự đoán cảm xúc từ ảnh khuôn mặt] 
def emotion_predict_thread():
    while True:
        face_img = emotion_queue.get()
        if face_img is None:
            break
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48)) / 255.0
            input_img = resized.reshape(1, 48, 48, 1)
            pred = emotion_model.predict(input_img, verbose=0)
            emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            emotion_result_queue.put(emotions[np.argmax(pred)])
        except:
            emotion_result_queue.put("Neutral")

# [7. Hàm dự đoán tuổi] 
def predict_age(face_img):
    resized = cv2.resize(face_img, (100, 100)) / 255.0
    input_img = resized.reshape(1, 100, 100, 3)
    age_pred, gender_pred = gender_model.predict(input_img, verbose=0)
    age = int(np.round(age_pred[0][0]))
    age_range = f"{max(age - 3, 0)} - {age + 3}"
    gender = "Male" if gender_pred[0][0] > 0.5 else "Female"
    return age_range, gender

# [7. Hàm dự đoán giới tính] 
def predict_gender(face_img):

    try: 
        if face_img.size == 0:
            return "Unknown"
        # Thay đổi kích thước ảnh 'face_img' thành 64x64 pixels và chuẩn hóa giá trị pixel về khoảng [0, 1] bằng cách chia cho 255
        resized = cv2.resize(face_img, (64, 64)) / 255.0

        # Thay đổi hình dạng (reshape) của ảnh 'resized' thành một mảng 4 chiều với kích thước (1, 100, 100, 3)
        # - 1 là kích thước batch (1 ảnh trong một batch)
        # - 100x100 là chiều cao và chiều rộng của ảnh
        # - 3 là số kênh màu (RGB)
        input_img = resized.reshape(1, 64, 64, 3)

        # verbose = 0, nghĩa là không in bất kỳ thông tin nào ra trong quá trình dự đoán
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



# [8. Khởi động các luồng xử lý đa nhiệm] 
threading.Thread(target=speech_loop, daemon=True).start()
threading.Thread(target=eye_predict_thread, daemon=True).start()
threading.Thread(target=emotion_predict_thread, daemon=True).start()
threading.Thread(target=gender_predict_thread, daemon=True).start()


# [9. Bắt đầu camera và xử lý khung hình chính] 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_emotion_time = 0
emotion_interval = 1.0

last_gender_time = 0
last_age_time = 0
age_interval = 5.0
gender_interval = 5.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ih, iw, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            x_coords = [int(pt.x * iw) for pt in face_landmarks.landmark]
            y_coords = [int(pt.y * ih) for pt in face_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            face_crop = frame[y_min:y_max, x_min:x_max]

            def crop_eye(indices):
                xs = [int(face_landmarks.landmark[i].x * iw) for i in indices]
                ys = [int(face_landmarks.landmark[i].y * ih) for i in indices]
                x_min_crop, x_max_crop = max(0, min(xs)-15), min(iw, max(xs)+15)
                y_min_crop, y_max_crop = max(0, min(ys)-15), min(ih, max(ys)+15)
                return frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

            left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 246]
            right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 466]
            left_eye = crop_eye(left_eye_indices)
            right_eye = crop_eye(right_eye_indices)

            if not eye_queue.full():
                eye_queue.put((left_eye, right_eye))

            if (time.time() - last_emotion_time) > emotion_interval:
                last_emotion_time = time.time()
                if not emotion_queue.full():
                    emotion_queue.put(face_crop)

            if (time.time() - last_age_time) > age_interval:
                current_age = predict_age(face_crop)
                last_age_time = time.time()
                

            if (time.time() - last_gender_time) > gender_interval:
                last_gender_time = time.time()
                if not gender_queue.full():
                    gender_queue.put(face_crop)

    # [10. Nhận kết quả và cập nhật trạng thái hệ thống] 
    handle_eye_state()  # Chỉ gọi hàm đã tối ưu

    if not emotion_result_queue.empty():
        current_emotion = emotion_result_queue.get()

    if not gender_result_queue.empty():
        current_gender = gender_result_queue.get()

    # [11. Hiển thị lên khung hình]
    cv2.putText(frame, f"Eye: {current_eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Emotion: {current_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if current_age and current_gender:
        cv2.putText(frame, f"Age: {current_age}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Gender: {current_gender}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

    cv2.imshow("Eye + Emotion + Age + Gender", frame)

    # Nhấn 'q' để thoát khỏi chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đã thoát khỏi While


# [12. Dọn dẹp tài nguyên] 
cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)
eye_queue.put(None)
emotion_queue.put(None)
gender_queue.put(None)
