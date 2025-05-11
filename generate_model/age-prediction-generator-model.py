import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from utils.process_age_dataset import process_age_dataset  # Hàm xử lý dữ liệu tuổi

# Lấy tập dữ liệu đã được xử lí (tuổi)
X_train, X_test, y_train, y_test = process_age_dataset()

def generate_model():
    model = Sequential()

    # Lớp tích chập đầu tiên: 32 filters, kernel 3x3, activation ReLU, padding 'same'
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
    model.add(BatchNormalization())  # BatchNormalization
    model.add(MaxPooling2D((2, 2)))

    # Lớp tích chập thứ hai: 64 filters, kernel 3x3
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())  # BatchNormalization
    model.add(MaxPooling2D((2, 2)))

    # Lớp tích chập thứ ba: 128 filters, kernel 3x3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())  # BatchNormalization
    model.add(MaxPooling2D((2, 2)))

    # Làm phẳng ma trận đặc trưng và thêm các lớp Dense
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # L2 Regularization
    model.add(Dropout(0.5))  # Dropout để tránh overfitting

    # Output: 1 neuron, activation linear cho hồi quy
    model.add(Dense(1, activation='linear'))

    # Compile với loss MAE và metric MAE
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

    return model

model = generate_model()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Đánh giá mô hình trên tập test
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Squared Error trên tập test: {loss:.4f}')
print(f'Mean Absolute Error trên tập test: {mae:.4f}')

# Lưu mô hình dự đoán tuổi
model.save("model/age_model.h5")
