import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

trainY_cat = to_categorical(trainY, 10)
testY_cat = to_categorical(testY, 10)


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")  
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(trainX, trainY_cat, 
                    validation_data=(testX, testY_cat),
                    epochs=10, batch_size=128, verbose=2)

loss, acc = model.evaluate(testX, testY_cat, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()


predictions = model.predict(testX[:20])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 8))
for i in range(20):
    plt.subplot(5, 4, i+1)
    plt.imshow(testX[i], cmap="gray")
    plt.axis("off")
    color = "green" if predicted_labels[i] == testY[i] else "red"
    plt.title(f"T:{testY[i]} P:{predicted_labels[i]}", fontsize=10, color=color)

plt.tight_layout()
plt.show()
