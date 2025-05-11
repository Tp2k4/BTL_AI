import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from utils.process_gender_dataset import process_gender_dataset

#Lấy tập dữ liệu đã được xử lí
X_train, X_test, y_train, y_test = process_gender_dataset()

#Khởi tạo mô hình
def generate_model():
    # Xây dựng mô hình CNN từ đầu
    model = Sequential()

    # Lớp tích chập đầu tiên: 32 filters, kernel 3x3, activation ReLU, padding 'same'
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))  # Lớp pooling 2x2

    # Lớp tích chập thứ hai: 64 filters, kernel 3x3
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Lớp tích chập thứ ba: 128 filters, kernel 3x3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Làm phẳng ma trận đặc trưng và thêm các lớp Dense
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout để tránh overfitting
    model.add(Dense(1, activation='sigmoid'))  # Output sigmoid cho phân loại nhị phân

    # Biên dịch mô hình với optimizer Adam và hàm mất mát binary_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Huấn luyện mô hình
model = generate_model()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Đánh giá mô hình trên tập test
loss, acc = model.evaluate(X_test, y_test)
print('Độ chính xác trên tập test:', acc)

model.save("model/gender_model.h5")