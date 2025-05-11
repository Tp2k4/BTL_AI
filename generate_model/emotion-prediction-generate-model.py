import os  
import numpy as np  
import cv2  
import matplotlib.pyplot as plt  
import seaborn as sns  
from keras.models import Sequential  
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.optimizers import AdamW  
from tensorflow.keras.callbacks import EarlyStopping  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Đường dẫn dataset
base_path = "dataset/archive/CK+48"

# Thiết lập thông số  
img_width, img_height = 48, 48  
batch_size = 32
num_classes = len([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

# Tạo ImageDataGenerator cho train và test  
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Chia 20% dữ liệu cho validation
)

# Tạo train_generator  
train_generator = datagen.flow_from_directory(  
    base_path,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    color_mode='grayscale',  
    class_mode='categorical',
    subset='training'
)  

# Tạo validation_generator  
validation_generator = datagen.flow_from_directory(  
    base_path,  
    target_size=(img_width, img_height),  
    batch_size=batch_size,  
    color_mode='grayscale',  
    class_mode='categorical',
    subset='validation'
)  

# Xây dựng mô hình CNN  
model = Sequential([  
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),  
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  
    
    Conv2D(64, (3, 3), activation='relu'),  
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu'),  
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  
    Dropout(0.4),
    
    Flatten(),  
    Dense(256, activation='relu'),  
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')  
])  

# Biên dịch mô hình với AdamW optimizer  
model.compile(optimizer=AdamW(learning_rate=0.0001),  
              loss='categorical_crossentropy',  
              metrics=['accuracy'])  

# Thêm EarlyStopping để tránh overfitting  
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình  
history = model.fit(  
    train_generator,  
    validation_data=validation_generator,  
    epochs=50,  
    callbacks=[early_stopping],
    verbose=1
)  

# Đánh giá mô hình  
y_true = validation_generator.classes
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)

# Hiển thị báo cáo phân loại chi tiết  
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_labels)
print("\nBáo cáo phân loại chi tiết:")
print(report)

# Tính độ chính xác tổng thể
accuracy = accuracy_score(y_true, y_pred)
print(f"\nĐộ chính xác tổng thể: {accuracy * 100:.2f}%")

# Ma trận nhầm lẫn  
conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')  
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Ma trận nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.show()

# Lưu mô hình  
model.save('model\emotion_model.keras')
print("\nMô hình đã được lưu thành công!")