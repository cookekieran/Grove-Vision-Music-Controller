import tensorflow as tf
import numpy as np
import glob
import cv2

loaded = tf.saved_model.load("tf_model")

concrete_func = None

for fn in loaded.signatures.values():
    concrete_func = fn
    break

if concrete_func is None:
    def wrapped(x):
        return loaded(x)
    
    concrete_func = tf.function(wrapped).get_concrete_function(
        tf.TensorSpec([1, 96, 96, 1], tf.float32)
    )

tf.saved_model.save(
    loaded,
    "tf_model_fixed",
    signatures={"serving_default": concrete_func}
)

print("fixed model signature")


converter = tf.lite.TFLiteConverter.from_saved_model("tf_model_fixed")

converter.optimizations = [tf.lite.Optimize.DEFAULT]

IMG_SIZE = (96, 96)

data_files = glob.glob("data/training/*/*.jpg")
np.random.shuffle(data_files)

# sample data
def representative_dataset():
    for file_path in data_files[:200]:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, IMG_SIZE)

        img = img.astype(np.float32) / 255.0 # normalisation

        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        yield [img]

converter.representative_dataset = representative_dataset

# compress to int8
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("model_int8.tflite created successfully.")