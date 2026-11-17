from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from s4_2_2_classifier_utils_tflite import classify, class_names
import requests

app = Flask(__name__)

def consultar_api_flask(estado_hoja: str):
    """Llama al API Flask enviando el texto de la enfermedad detectada"""
    try:
        url = "http://127.0.0.1:5000/diagnostico"
        params = {"estado_hoja": estado_hoja}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("posible_enfermedad", "Sin respuesta")
        else:
            return f"Error en API: {response.status_code}"
    except Exception as e:
        return f"Error de conexión: {e}"


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    all_scores = None
    image_data = None
    diagnostico = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convertir a base64 para mostrarla
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            image_data = jpg_as_text

            class_name, confidence, prediction = classify(image_rgb)
            scores = [(class_names[i], float(prediction[i]) * 100) for i in range(len(class_names))]

            result = f"{class_name} - {confidence:.2f}%"
            all_scores = scores
            
            # 🔹 Llamar API externa
            diagnostico = consultar_api_flask(class_name)
            print(diagnostico)

    return render_template('index_s4_2_1_b.html', result=result, scores=all_scores, image_data=image_data, diagnostico=diagnostico)

if __name__ == '__main__':
   #app.run(debug=True)
    app.run(host="0.0.0.0", port=5001, debug=True)

