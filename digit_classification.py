import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

x_train_mnist = x_train_mnist.astype("float32") / 255.0
x_test_mnist = x_test_mnist.astype("float32") / 255.0

y_train_mnist = to_categorical(y_train_mnist, 10)
y_test_mnist = to_categorical(y_test_mnist, 10)


data = np.load("mnist_custom.npz")

x_train_custom = data["trainX"].astype("float32") / 255.0
y_train_custom = to_categorical(data["trainY"], 10)

x_test_custom = data["testX"].astype("float32") / 255.0
y_test_custom = to_categorical(data["testY"], 10)

x_train = np.concatenate((x_train_mnist, x_train_custom), axis=0)
y_train = np.concatenate((y_train_mnist, y_train_custom), axis=0)

x_test = np.concatenate((x_test_mnist, x_test_custom), axis=0)
y_test = np.concatenate((y_test_mnist, y_test_custom), axis=0)

print("Dataset Summary:")
print(f"MNIST Train: {x_train_mnist.shape[0]} samples")
print(f"Custom Train: {x_train_custom.shape[0]} samples")
print(f"Total Train: {x_train.shape[0]} samples")

print(f"MNIST Test: {x_test_mnist.shape[0]} samples")
print(f"Custom Test: {x_test_custom.shape[0]} samples")
print(f"Combined Test: {x_test.shape[0]} samples")


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=128
)

print(f"\nValidation samples used: {x_test.shape[0]}")

print("\n--- Separate Evaluations ---")
loss_mnist, acc_mnist = model.evaluate(x_test_mnist, y_test_mnist, verbose=0)
print(f"MNIST Test Accuracy: {acc_mnist*100:.2f}%")

loss_custom, acc_custom = model.evaluate(x_test_custom, y_test_custom, verbose=0)
print(f"Custom Test Accuracy: {acc_custom*100:.2f}%")

loss_combined, acc_combined = model.evaluate(x_test, y_test, verbose=0)
print(f"Combined Test Accuracy: {acc_combined*100:.2f}%")

predictions = model.predict(x_test_custom[:25])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test_custom[:25], axis=1)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test_custom[i], cmap="gray")
    plt.title(f"T:{true_labels[i]} P:{predicted_labels[i]}")
    plt.axis("off")
plt.show()


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
