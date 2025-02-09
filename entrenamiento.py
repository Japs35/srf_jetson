import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
import os
import numpy as np

# Crear la ventana principal
root = tk.Tk()
root.title("Registro IES RFA")
root.geometry("1200x600")  # Tamaño de la ventana

# Crear un notebook para las pestañas
notebook = ttk.Notebook(root)

# Crear las tres pestañas
tab_captura = ttk.Frame(notebook)
tab_srf = ttk.Frame(notebook)
tab_entrenamiento = ttk.Frame(notebook)

# Agregar las pestañas al notebook
notebook.add(tab_captura, text="CAPTURA")
notebook.add(tab_srf, text="SRF")
notebook.add(tab_entrenamiento, text="ENTRENAMIENTO")

# Crear un label en la pestaña CAPTURA
label_captura = tk.Label(tab_captura, text="Registro de usuario", font=("Arial", 14))
label_captura.pack(pady=20)

# Variable para el nombre del usuario
nombre_usuario = tk.StringVar()

# Función para registrar al usuario
def registrar_usuario():
    global nombre_usuario
    nombre = nombre_usuario.get()
    if nombre:
        # Crear una carpeta con el nombre del usuario
        folder_path = os.path.join('usuarios', nombre)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Iniciar la captura de fotos
        capture_fotos(folder_path)
    else:
        messagebox.showerror("Error", "Por favor, ingrese un nombre")

# Función para capturar fotos desde la cámara
def capture_fotos(folder_path):
    cap = cv2.VideoCapture("rtsp://admin:123456abc@192.168.1.64:554/stream")
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara")
        return

    fotos_capturadas = 0
    while fotos_capturadas < 40:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el frame de la cámara")
            break

        # Mostrar la imagen en la ventana de tkinter
        cv2.imshow("Captura de imágenes", frame)

        # Esperar la tecla 'c' para capturar una foto
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            foto_nombre = os.path.join(folder_path, f"{fotos_capturadas + 1}.jpg")
            cv2.imwrite(foto_nombre, frame)
            fotos_capturadas += 1
            print(f"Foto {fotos_capturadas} capturada")

        # Si presionamos 'q' se cierra la cámara
        if key == ord('q'):
            break

    # Al terminar las 40 fotos
    messagebox.showinfo("Éxito", "Usuario registrado con éxito")
    cap.release()
    cv2.destroyAllWindows()

# Frame para el nuevo registro
frame_registro = ttk.Frame(tab_captura)
frame_registro.pack(pady=20)

# Etiqueta y campo de texto para el nombre del usuario
label_nombre = tk.Label(frame_registro, text="Ingrese el nombre completo del usuario:")
label_nombre.grid(row=0, column=0, padx=10, pady=10)
entry_nombre = tk.Entry(frame_registro, textvariable=nombre_usuario, font=("Arial", 12))
entry_nombre.grid(row=0, column=1, padx=10, pady=10)

# Botón para registrar al usuario
btn_registrar = ttk.Button(frame_registro, text="Nuevo Registro", command=registrar_usuario)
btn_registrar.grid(row=1, columnspan=2, pady=20)

# Crear un label en la pestaña SRF
label_srf = tk.Label(tab_srf, text="SRF", font=("Arial", 14))
label_srf.pack(pady=20)

# Función para el entrenamiento con Haar Cascade
def entrenar_modelo():
    # Mensaje de inicio de entrenamiento
    messagebox.showinfo("Entrenamiento", "Iniciando entrenamiento...")

    # Cargar el clasificador Haar Cascade para detección de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Crear barra de progreso
    progress = ttk.Progressbar(tab_entrenamiento, orient="horizontal", length=300, mode="indeterminate")
    progress.pack(pady=20)
    progress.start()

    # "Entrenamiento" simulado
    for i in range(1, 101):  # Simulamos 100 pasos de entrenamiento
        progress["value"] = i
        root.update_idletasks()  # Actualiza la interfaz gráfica
        cv2.waitKey(10)  # Pausa corta para simular el tiempo de entrenamiento

    progress.stop()  # Detener la barra de progreso
    messagebox.showinfo("Entrenamiento", "Entrenamiento completado")

# Frame para el entrenamiento
frame_entrenamiento = ttk.Frame(tab_entrenamiento)
frame_entrenamiento.pack(pady=20)

# Botón para iniciar el entrenamiento
btn_entrenar = ttk.Button(frame_entrenamiento, text="Iniciar Entrenamiento", command=entrenar_modelo)
btn_entrenar.pack(pady=20)

# Función para el reconocimiento en tiempo real (aparece al ingresar a la pestaña SRF)
def iniciar_reconocimiento():
    # Abrir las cámaras IP usando las URLs proporcionadas
    IP_CAMERA1_URL = "rtsp://admin:123456abc@192.168.1.64:554/stream"
    #IP_CAMERA2_URL = "rtsp://admin:123456abc@192.168.1.65:554/stream"

    cap1 = cv2.VideoCapture(IP_CAMERA1_URL)  # Cámara 1 (IP)
    #cap2 = cv2.VideoCapture(IP_CAMERA2_URL)  # Cámara 2 (IP)

    if not cap1.isOpened(): #or not cap2.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a las cámaras")
        return

    # Cargar el clasificador Haar Cascade para detección de caras
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret1, frame1 = cap1.read()
        #ret2, frame2 = cap2.read()

        if not ret1: #or not ret2:
            messagebox.showerror("Error", "No se pudo leer el frame de las cámaras")
            break

        # Convertir a escala de grises para detección
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detectar caras en ambas cámaras
        faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar rectángulos alrededor de las caras detectadas
        for (x, y, w, h) in faces1:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #for (x, y, w, h) in faces2:
            #cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar las imágenes de las cámaras en dos ventanas
        cv2.imshow("Cámara 1", frame1)
        #cv2.imshow("Cámara 2", frame2)

        # Terminar con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    #cap2.release()
    cv2.destroyAllWindows()

# Empaquetar el notebook (se convierte en la interfaz principal)
notebook.pack(expand=1, fill="both")

# Llamar a la función para iniciar el reconocimiento en SRF solo cuando se ingresa a esa pestaña
notebook.select(tab_srf)
btn_reconocer = ttk.Button(tab_srf, text="Iniciar Reconocimiento", command=iniciar_reconocimiento)
btn_reconocer.pack(pady=20)

# Iniciar el bucle de la interfaz
root.mainloop()
