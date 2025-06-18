import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset path
train_dir = 'dataset/train'
valid_dir = 'dataset/valid'

# Preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
valid_gen = ImageDataGenerator(rescale=1./255)
=
train_data = train_gen.flow_from_directory(train_dir, target_size=(150, 150), class_mode='categorical')
valid_data = valid_gen.flow_from_directory(valid_dir, target_size=(150, 150), class_mode='categorical')

# Model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
model.fit(train_data, epochs=10, validation_data=valid_data)
# Evaluasi model pada data validasi
loss, acc = model.evaluate(valid_data)
print(f"Akurasi validasi akhir: {acc * 100:.2f}%")

# Simpan model
model.save("model_daging.h5")