import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from tkinter import *
from tkinter import messagebox
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Função para carregar imagens de referência
def load_reference_images(path='trabalho_ia/known_faces'):
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Pasta "{path}" criada. Adicione imagens de referência nessa pasta.')
        return known_face_encodings, known_face_names

    for file_name in os.listdir(path):
        if file_name.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(path, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file_name)[0])

    return known_face_encodings, known_face_names

# Função para marcar a presença e enviar e-mail
def mark_attendance(name):
    email_dict = {
        # 'name': 'email@example.com', Adicione os nomes e e-mails dos alunos aqui
        'Aluno1': 'danilo.oliveira@somosicev.com',
        'Aluno2': 'davi.lopes@somosicev.com',
        'Aluno3': 'whalisson@hotmail.com'
    }
    
    with open('trabalho_ia/attendance.csv', 'r+') as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]

        if name not in names:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dt_string}')
            
            if name in email_dict:
                send_email(email_dict[name], name)

# Função para enviar e-mail
def send_email(receiver_email, name):
    sender_email = 'danilo.cdo@grupocev.com'
    sender_password = 'sua_senha'
    subject = 'Confirmação de Presença'
    body = f'Olá {name},\n\nSua presença foi marcada com sucesso em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.\n\nAtenciosamente,\nEquipe ICEV'

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.example.com', 587)  # Atualize com o servidor SMTP e porta corretos
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f'Email enviado para {receiver_email}')
    except Exception as e:
        print(f'Erro ao enviar email: {e}')

# Função principal do reconhecimento facial
def recognize_faces(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)

    # Ajuste do brilho da câmera do Mac.
    video_capture.set(cv2.CAP_PROP_BRIGHTNESS, 150)
    video_capture.set(cv2.CAP_PROP_CONTRAST, 150)
    video_capture.set(cv2.CAP_PROP_EXPOSURE, -4)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro ao acessar a câmera")
            break

        # Converter o frame capturado para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Não é aluno ICEV!"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            mark_attendance(name)

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Interface do app de reconhecimento
class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento Facial - Frequência Escolar")

        self.start_button = Button(root, text="Iniciar Reconhecimento", command=self.start_recognition)
        self.start_button.pack(pady=20)

        self.exit_button = Button(root, text="Sair do app", command=root.quit)
        self.exit_button.pack(pady=20)

    def start_recognition(self):
        known_face_encodings, known_face_names = load_reference_images()
        if not known_face_encodings:
            messagebox.showwarning("Aviso", "Nenhuma imagem de referência encontrada. Adicione imagens na pasta 'known_faces'.")
            return
        recognize_faces(known_face_encodings, known_face_names)

if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    root.mainloop()

print(os.getcwd())