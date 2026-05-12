import os
import tensorflow as tf
import numpy as np

MODEL_PATH = "Backend/neoscan_eye_FINAL_effb4.keras"
try:
    print("TensorFlow Version:", tf.__version__)
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("SUCCESS: Eye model loaded!")
        # Test with dummy data
        dummy = np.random.rand(1, 380, 380, 3).astype("float32")
        pred = model.predict(dummy, verbose=0)
        print("Prediction shape:", pred.shape)
    else:
        print("ERROR: Model not found at", MODEL_PATH)
except Exception as e:
    print("ERROR:", e)
