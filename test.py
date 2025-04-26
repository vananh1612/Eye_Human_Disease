from tkinter import _test
from tensorflow import keras

model = keras.models.load_model("Trained_Eye_disease_model.keras")

loss, accuracy = model.evaluate(_test, _test)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")
