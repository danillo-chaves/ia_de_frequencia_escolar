import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import Label, Button, Entry
from PIL import Image, ImageTk
from datetime import datetime


# Diretório contendo imagens de rostos conhecidos
KNOWN_FACES_DIR = 'known_faces2'
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
                continue
            image = cv2.resize(image, FACE_SIZE)
            faces.append(image.flatten())
            labels.append(label)
    if not faces or not labels:
        raise ValueError("Nenhuma imagem ou label foi carregada.")
    return np.array(faces), np.array(labels)

# Função para salvar nova face
def save_new_face(image, name):
    new_face_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(new_face_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(new_face_dir, f"{timestamp}.jpg")
    cv2.imwrite(file_path, image)
    print(f"Imagem salva em: {file_path}")

# Carregar imagens de rostos conhecidos
faces, labels = load_known_faces(KNOWN_FACES_DIR)
if faces.size == 0:
    raise ValueError("Nenhuma imagem de rosto conhecida encontrada.")

# Verificar se faces e labels não estão vazios
if faces.size == 0 or labels.size == 0:
    raise ValueError("Os dados de faces ou labels estão vazios.")

# Ajustar n_neighbors dinamicamente
n_neighbors = min(3, len(labels))
if n_neighbors < 1:
    raise ValueError("Número insuficiente de amostras para treinamento.")

# Treinar o modelo de reconhecimento
model = KNeighborsClassifier(n_neighbors=n_neighbors)
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
        
        try:
            # Reconhecer o rosto
            label = model.predict(face_resized)[0]
        except ValueError as e:
            print(f"Erro ao prever: {e}")
            continue
        
        # Desenhar o retângulo e o label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Convertendo o frame para exibição no Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Atualizar a imagem na interface Tkinter
    display_label.imgtk = imgtk
    display_label.configure(image=imgtk)

    # Chamar a função update_frame após um curto intervalo
    root.after(10, update_frame)

# Função para capturar e salvar nova imagem
def capture_new_image():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")
        return
    
    # Converter para grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(detected_faces) == 0:
        print("Nenhum rosto detectado.")
        return
    
    x, y, w, h = detected_faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, FACE_SIZE)

    # Solicitar o nome
    new_name = name_entry.get()
    if not new_name:
        print("Por favor, insira um nome.")
        return
    
    save_new_face(face_resized, new_name)
    print(f"Nova face para {new_name} capturada e salva.")
    name_entry.delete(0, tk.END)  # Limpar a entrada do nome

# Configuração da interface gráfica
display_label = Label(root)
display_label.pack()

name_entry = Entry(root)
name_entry.pack()
name_entry.insert(0, "Digite o nome")

capture_button = Button(root, text="Capturar e Salvar", command=capture_new_image)
capture_button.pack(side="left")

start_button = Button(root, text="Iniciar Reconhecimento", command=update_frame)
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
