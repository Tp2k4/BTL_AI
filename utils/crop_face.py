import mediapipe as mp
import numpy as np
import cv2
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def crop_face(img):
    ih, iw, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_result = face_mesh.process(img_rgb)

    if img_result.multi_face_landmarks:
        face_landmarks = img_result.multi_face_landmarks[0]

        x_coords = [int(pt.x * iw) for pt in face_landmarks.landmark]
        y_coords = [int(pt.y * ih) for pt in face_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        face_crop = img[y_min:y_max, x_min:x_max]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            return None

        face_crop = cv2.resize(face_crop, (64, 64))
        
        return face_crop, x_min, x_max, y_min, y_max
        
    return None