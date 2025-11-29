# Install required packages
# !pip install tensorflow emnist matplotlib

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Option 1: MNIST (digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Option 2: EMNIST (letters) - Uncomment if using EMNIST
# x_train, y_train = extract_training_samples('letters')
# x_test, y_test = extract_test_samples('letters')

# -------------------------------
# 2. Preprocess Data
# -------------------------------
# Reshape to (num_samples, 28, 28, 1) for CNN
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# -------------------------------
# 3. Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 4. Train Model
# -------------------------------
history = model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=128)

# -------------------------------
# 5. Evaluate Model
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# -------------------------------
# 6. Predict Example
# -------------------------------
index = 10
plt.imshow(x_test[index].reshape(28,28), cmap='gray')
plt.title(f"True Label: {np.argmax(y_test[index])}")
plt.show()

pred = model.predict(np.expand_dims(x_test[index], axis=0))
print(f"Predicted Label: {np.argmax(pred)}")
