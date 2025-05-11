import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers


data = []
labels = []

for label, category in enumerate(['closed_eye', 'open_eye']):
    folder = os.path.join('dataset\driver-drowsiness-detection', category)
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

print("Labeling okok")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
]) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)
model.save("model\eye_state_model.h5")



