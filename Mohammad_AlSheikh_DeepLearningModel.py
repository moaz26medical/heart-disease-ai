import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU,
)
import os

fixed_len = 200
num_classes = 3
model = Sequential(
    [
        Conv1D(200, 7, input_shape=(fixed_len, 1)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 5),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 5),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(64),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
save_dir = r"C:\Users\Mohammed\Desktop\DL-CIDEA_model\SavedModels"
os.makedirs(save_dir, exist_ok=True)
h5_path = os.path.join(save_dir, "ecg_cnn_full_model.h5")
model.save(h5_path)
model_loaded = tf.keras.models.load_model(h5_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model_loaded)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
tflite_path = os.path.join(save_dir, "ecg_cnn_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
