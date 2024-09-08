import cv2
import numpy as np
from kafka import KafkaConsumer
from prediction.utils import Pose_estimator
import tempfile
import os
import tkinter as tk
import customtkinter as ck
from PIL import Image, ImageTk

# Initialiser Pose_estimator
pose_estimator = Pose_estimator(model_path='deadlift.pkl')

# Configuration Kafka
kafka_server = 'localhost:9092'
input_topic = 'traited1'

# Consommer les frames du topic Kafka
consumer = KafkaConsumer(
    input_topic,
    bootstrap_servers=kafka_server,
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    group_id='pose_estimation_group',
    value_deserializer=lambda m: m  # Dé-sérialiser les messages en bytes
)

# Créer un répertoire temporaire pour stocker les frames traitées
temp_dir = tempfile.mkdtemp()

# Liste pour stocker les frames traitées
frames = []
class_up_down = []

# Fonction pour mettre à jour l'interface Tkinter
def update_interface(frame, stage):
    cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    imgtk = ImageTk.PhotoImage(image=pil_image)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    classBox.configure(text=stage)

# Fonction pour traiter les frames et mettre à jour l'interface
def process_and_update_interface():
    global counter
    for message in consumer:
        img_bytes = message.value
        np_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is not None:
            processed_frame, stage = pose_estimator.process_image(frame, None, None)
            frames.append(processed_frame)
            class_up_down.append(stage)
            
            update_interface(processed_frame, stage)

        if len(frames) >= 376:  # Nombre de frames avant sauvegarde
            break

    consumer.close()

# Interface Tkinter
window = tk.Tk()
window.geometry("480x700")
window.title("Swoleboi") 
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE') 
counterLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS') 
probLabel  = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB') 
classBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0') 
counterBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0') 
probBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0') 

def reset_counter(): 
    global counter
    counter = 0 

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

# Exécuter le traitement des frames et mettre à jour l'interface
process_and_update_interface()

# Démarrer la boucle principale Tkinter
window.mainloop()
