# train_eye_model.py
import pickle
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

# Load dữ liệu đã xử lý
with open('dataset/processed_eye_data.pkl', 'rb') as f:
    X, y = pickle.load(f)

print("Data loaded. Starting training...")

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình CNN

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])


# Biên dịch và huấn luyện
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)

# Lưu mô hình
model.save("eye_state_model.h5")
print("Model saved to model/eye_state_model.h5")
