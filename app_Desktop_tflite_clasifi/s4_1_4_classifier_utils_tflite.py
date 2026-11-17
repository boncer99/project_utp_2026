# classifier_utils_tflite.py
import cv2
import numpy as np
import tensorflow as tf

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Leer labels
class_names = [line.strip() for line in open("labels.txt", "r")]

def preprocess(image):
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    #img = np.asarray(img, dtype=np.float32)
    img = np.asarray(img, dtype=np.uint8)
    #img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    return img

def classify(image):
    processed = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

     # Corregir escala si los valores son 0-255
    if output_data.max() > 1.0:
        output_data = output_data / 255.0

    index = np.argmax(output_data)
    class_name = class_names[index]
    confidence = float(output_data[index]) * 100
    return class_name, confidence, output_data
