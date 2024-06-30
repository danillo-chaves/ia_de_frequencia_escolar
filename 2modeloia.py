import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import Label, Button, Entry
from PIL import Image, ImageTk
from datetime import datetime
import pickle
import logging

# Configuração do log
logging.basicConfig(
    filename='2modelo.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Função para registrar eventos no log
def log_event(event):
    logging.info(event)

# Diretório contendo imagens de rostos conhecidos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = 'known_faces2'
MODEL_DIR = 'models/modelo_treinado.pkl'

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
                log_event(f"Falha ao carregar a imagem: {path}")
                continue
            image = cv2.resize(image, FACE_SIZE)
            faces.append(image.flatten())
            labels.append(label)
    log_event(f"Faces e labels carregados de {directory}")
    return np.array(faces), np.array(labels)

# Função para salvar nova face
def save_new_face(image, name):
    new_face_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(new_face_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(new_face_dir, f"{timestamp}.jpg")
    cv2.imwrite(file_path, image)
    log_event(f"Nova face salva para {name} em {file_path}")
    retrain_model()

# Função para re-treinar o modelo
def retrain_model():
    global model, faces, labels
    faces, labels = load_known_faces(KNOWN_FACES_DIR)
    n_neighbors = min(3, len(labels))
    if n_neighbors < 1:
        log_event("Número insuficiente de amostras para treinamento.")
        print("Número insuficiente de amostras para treinamento.")
        return
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(faces, labels)
    log_event("Modelo re-treinado com sucesso.")
    print("Modelo re-treinado com sucesso.")
    save_model(model, MODEL_DIR)

# Função para salvar o modelo treinado
def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    log_event(f"Modelo salvo em: {path}")
    print(f"Modelo salvo em: {path}")

# Função para carregar o modelo treinado
def load_model(path):
    with open(path, 'rb') as f:
        log_event(f"Modelo carregado de: {path}")
        return pickle.load(f)

# Inicializar o Tkinter
root = tk.Tk()
root.title("Reconhecimento Facial")
log_event("Interface Tkinter inicializada.")

# Iniciar a webcam
cap = cv2.VideoCapture(0)
log_event("Webcam iniciada.")

# Carregar ou criar o modelo
model_path = 'knn_model.pkl'
if os.path.exists(model_path):
    model = load_model(model_path)
    log_event("Modelo carregado com sucesso.")
else:
    faces, labels = load_known_faces(KNOWN_FACES_DIR)
    if faces.size == 0 or labels.size == 0:
        log_event("Os dados de faces ou labels estão vazios.")
        raise ValueError("Os dados de faces ou labels estão vazios.")

    n_neighbors = min(3, len(labels))
    if n_neighbors < 1:
        log_event("Número insuficiente de amostras para treinamento.")
        raise ValueError("Número insuficiente de amostras para treinamento.")
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(faces, labels)
    log_event("Modelo treinado com sucesso.")
    save_model(model, MODEL_DIR)

# Função para exibir o frame da webcam na GUI
def update_frame():
    ret, frame = cap.read()
    if not ret:
        log_event("Falha ao capturar o frame da webcam.")
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
            proba = model.predict_proba(face_resized)
            confidence = np.max(proba) * 100  # Obter a confiança da previsão
        except ValueError as e:
            log_event(f"Erro ao prever: {e}")
            continue
        
        # Desenhar o retângulo e o label com a confiança
        text = f"{label} ({confidence:.2f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        log_event(f"Rosto reconhecido: {text} em ({x}, {y}, {w}, {h})")

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
        log_event("Erro ao capturar a imagem.")
        print("Erro ao capturar a imagem.")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(detected_faces) == 0:
        log_event("Nenhum rosto detectado.")
        print("Nenhum rosto detectado.")
        return
    
    x, y, w, h = detected_faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, FACE_SIZE)

    new_name = name_entry.get()
    if not new_name:
        log_event("Nome não fornecido ao tentar capturar nova face.")
        print("Por favor, insira um nome.")
        return
    
    save_new_face(face_resized, new_name)
    log_event(f"Nova face para {new_name} capturada e salva.")
    print(f"Nova face para {new_name} capturada e salva.")
    name_entry.delete(0, tk.END)  # Limpar a entrada do nome

def list_registered_names():
    if not os.path.exists(KNOWN_FACES_DIR):
        messagebox.showinfo("Info", "Nenhum rosto conhecido registrado.")
        return
    names = os.listdir(KNOWN_FACES_DIR)
    messagebox.showinfo("Rostos Registrados", "\n".join(names))

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

log_event("Interface gráfica configurada.")

# Iniciar o loop principal do Tkinter
root.mainloop()
log_event("Loop principal do Tkinter iniciado.")

# Liberar a webcam e fechar janelas
cap.release()
cv2.destroyAllWindows()
