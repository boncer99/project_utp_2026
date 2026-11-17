# app_gui.py
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import requests  # Para consumir la API Flask

from s4_1_4_classifier_utils_tflite import classify, class_names

# Matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

window = tk.Tk()
window.title("Leaf Classification - GUI con gráfico ordenado")
window.geometry("950x650")

camera = None
streaming = False

# Crear figure sin pop-up
fig = Figure(figsize=(4.5, 3))
ax = fig.add_subplot(111)

def update_bar_chart(scores):
    ax.clear()
    ax.bar(class_names, scores * 100)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelsize=5)  # <-- Cambia 10 por el tamaño que quieras
    ax.set_ylabel("Confidence (%)")
    
    fig.tight_layout()
    canvas_chart.draw()

'''
Si la API responde bien (200 OK), convierte la respuesta en JSON.
Extrae el campo "posible_enfermedad" de la respuesta de Flask.
Si no existe, devuelve "Sin respuesta".
'''
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

def classify_and_display(image):
    class_name, confidence, full_pred = classify(image)
    result_label.config(text=f"Class: {class_name} - {confidence:.2f}%")

    all_predictions_text.config(state='normal')
    all_predictions_text.delete(1.0, tk.END)
    for name, score in zip(class_names, full_pred):
        all_predictions_text.insert(tk.END, f"{name}: {score*100:.2f}%\n")
    all_predictions_text.config(state='disabled')

    update_bar_chart(full_pred)
    
    print(class_name)
    # 🔹 Aquí llamamos a la API Flask con el nombre de la clase detectada
    descripcion = consultar_api_flask(class_name)
    # Mostramos el diagnóstico en la caja de texto especial
    api_response_text.config(state='normal')
    api_response_text.delete(1.0, tk.END)
    api_response_text.insert(tk.END, descripcion)
    api_response_text.config(state='disabled')
    
    
    

def load_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    loaded_image = image_rgb.copy()
    show_on_canvas(image_rgb)
    classify_and_display(image_rgb)

def show_on_canvas(image_rgb):
    img_pil = Image.fromarray(image_rgb)
    img_pil = img_pil.resize((400, 300))
    tk_img = ImageTk.PhotoImage(img_pil)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    canvas.image = tk_img

def start_camera():
    global camera, streaming
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    streaming = True
    update_frame()

def stop_camera():
    global streaming
    streaming = False
    if camera:
        camera.release()

def update_frame():
    if streaming:
        ret, frame = camera.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            show_on_canvas(rgb)
            classify_and_display(rgb)
        window.after(50, update_frame)

# ------------------- LAYOUT ORGANIZADO ------------------- #

left_frame = tk.Frame(window)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

right_frame = tk.Frame(window)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Canvas (imagen) en izquierda
canvas = tk.Canvas(left_frame, width=400, height=300, bg="gray")
canvas.pack()

# Resultado principal
result_label = tk.Label(left_frame, text="Result:", font=("Arial", 16))
result_label.pack(pady=(10,0))

# Caja con todos los porcentajes
all_predictions_text = tk.Text(left_frame, height=6, width=50, font=("Arial",12))
all_predictions_text.pack(pady=5)
all_predictions_text.config(state='disabled')

# Botones debajo
btn_frame = tk.Frame(left_frame)
btn_frame.pack(pady=10)

btn_load = tk.Button(btn_frame, text="Upload Image", command=load_image)
btn_load.grid(row=0, column=0, padx=5)

btn_start = tk.Button(btn_frame, text="Start Cam", command=start_camera)
btn_start.grid(row=0, column=1, padx=5)

btn_stop = tk.Button(btn_frame, text="Stop Cam", command=stop_camera)
btn_stop.grid(row=0, column=2, padx=5)

# Gráfico matplotlib a la derecha
canvas_chart = FigureCanvasTkAgg(fig, master=right_frame)
canvas_chart.get_tk_widget().pack()


# Caja de texto para respuesta del API
api_response_label = tk.Label(right_frame, text="Descripción desde API Flask:", font=("Arial", 14))
api_response_label.pack(pady=(15, 0))

api_response_text = tk.Text(right_frame, height=5, width=50, font=("Arial", 12), wrap="word")
api_response_text.pack(pady=5)
api_response_text.config(state='disabled')

window.mainloop()

