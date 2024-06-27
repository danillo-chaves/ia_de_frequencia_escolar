import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Diretório contendo imagens de rostos conhecidos
KNOWN_FACES_DIR = 'known_faces'
# Parâmetros
FACE_SIZE = (150, 150)

# Função para carregar imagens e labels
def load_known_faces(directory):
    faces = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            path = os.path.join(label_dir, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Erro ao carregar imagem: {path}")
                continue
            image = cv2.resize(image, FACE_SIZE)
            faces.append(image.flatten())
            labels.append(label)
    return np.array(faces), np.array(labels)

# Carregar imagens de rostos conhecidos
faces, labels = load_known_faces(KNOWN_FACES_DIR)
if faces.size == 0:
    raise ValueError("Nenhuma imagem de rosto conhecida encontrada.")

# Treinar o modelo de reconhecimento
model = KNeighborsClassifier(n_neighbors=3)
model.fit(faces, labels)

print("Modelo treinado com sucesso.")

# Inicializar o Tkinter
root = tk.Tk()
root.title("Reconhecimento Facial")

# Iniciar a webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Não foi possível abrir a webcam.")


# Função para exibir o frame da webcam na GUI
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, FACE_SIZE).flatten()
        face_resized = face_resized.reshape(1, -1)  # Transforma em uma matriz 2D
        
        # Reconhecer o rosto
        label = model.predict(face_resized)[0]
        
        # Desenhar o retângulo e o label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convertendo o frame para exibição no Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, update_frame)

# Configuração da interface gráfica
label = Label(root)
label.pack()

start_button = Button(root, text="Iniciar", command=update_frame)
start_button.pack(side="left")

quit_button = Button(root, text="Sair", command=root.quit)
quit_button.pack(side="right")

# Função para liberar recursos ao sair
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Iniciar o loop principal do Tkinter
root.mainloop()
