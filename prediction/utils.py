# Classe de connexion à la base de données et traitement des images

from pymongo import MongoClient
import numpy as np
import cv2

import face_recognition

class Connexion_db:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', db_name='face_recognition_db'):
        # Connexion à MongoDB
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.students_collection = self.db['players']
        self.encodings_collection = self.db['encodings']

        # Charger les encodages depuis MongoDB
        print("Loading Encode Data from MongoDB...")
        encodeData = self.encodings_collection.find_one()
        self.encodeListKnown = np.array(encodeData['encodings'])
        self.studentIds = encodeData['student_ids']
        print("Encode Data Loaded")

    def extract_data(self):
        return self.encodeListKnown, self.studentIds

class Processing_image:
    def __init__(self):
        self.encodings_known = False  # Ajout pour vérifier si l'encodage est déjà fait

    def process_image(self, img, encodeListKnown, studentIds, process_faces=True):
        detected_info = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if process_faces or not self.encodings_known:
            # Effectuer la reconnaissance faciale seulement si nécessaire
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
                faceDis = face_recognition.face_distance(encodeListKnown, face_encoding)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    id = studentIds[matchIndex]
                    detected_info.append({
                        'id': id,
                        'bbox': (left, top, right - left, bottom - top)
                    })

            # Marquer comme déjà encodé après les premières frames
            self.encodings_known = True

        # Dessiner les informations détectées sur l'image
        for info in detected_info:
            if 'bbox' in info:
                cv2.rectangle(img, (info['bbox'][0], info['bbox'][1]),
                              (info['bbox'][0] + info['bbox'][2], info['bbox'][1] + info['bbox'][3]),
                              (0, 255, 0), 2)
                cv2.putText(img, f"ID: {info['id']}", (info['bbox'][0], info['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr, detected_info



import numpy as np
import face_recognition
from pymongo import MongoClient


import pandas as pd
import pickle
import mediapipe as mp
from landmarks import landmarks  # Assure-toi que le module landmarks est bien disponible



class Pose_estimator:
    def _init_(self, model_path='deadlift.pkl'):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        
        # Charger le modèle
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            print(f"Erreur : le fichier modèle {model_path} est introuvable.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")

    def process_image(self, img):
     
        current_stage = ''
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        # Détection des landmarks
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                self.mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10)
            )

            try:
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks)
                
                bodylang_prob = self.model.predict_proba(X)[0]
                bodylang_class = self.model.predict(X)[0]

                if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    current_stage = "down"
                elif bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    current_stage = "up"
            
            except Exception as e:
                print(f"Erreur lors de la prédiction : {e}")
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr,bodylang_class