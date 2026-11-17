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

'''
Con ese tipo de salida (“25500%” o 100% en lugares incorrectos), ya confirmamos claramente lo que está pasando:

🚨 Diagnóstico rápido
es típico cuando:
Se carga un modelo TFLite cuantizado que espera datos uint8 pero...
No se normaliza ni escala los valores correctamente
O bien la interpretación de la salida del intérprete es distinta a lo esperado (por ejemplo, el modelo devuelve valores entre 0 y 255 en lugar de 0 a 1).
Esto provoca que aparezcan números absurdos como 25500% (que equivale a valor ≈ 255 en bruto * 100).

✅ ¿Cómo se corrige?

Necesitamos escalar la salida si el modelo entrega valores de 0–255.
Posible solución directa:
Agregar una división por 255 en tu vector de predicción:

✅ Conclusión y recomendación

Quieres probar exactamente lo mismo que vas a correr en Android ✅
Pues en Android también tendrás que detectar si es quantized o float modelo.
Si seleccionas Quantized, debes normalizar en escala 0–255 y dividir por 255 para la salida → exactamente como te mostré.
Si no quieres lidiar con conversiones, usa el modelo TFLite en float32.
'''
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
