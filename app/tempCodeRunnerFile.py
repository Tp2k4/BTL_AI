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