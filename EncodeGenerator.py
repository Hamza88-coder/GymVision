import cv2
import face_recognition
import pickle
import os
from pymongo import MongoClient

# Connexion à MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
students_collection = db['players']
encodings_collection = db['encodings']

# Importation des images des étudiants
folderPath = "Images"
pathList = os.listdir(folderPath)
imgList = []
studentIDs = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentID = os.path.splitext(path)[0]
    studentIDs.append(studentID)

    # Ajouter l'image de l'étudiant à MongoDB
    with open(os.path.join(folderPath, path), 'rb') as image_file:
        encoded_image = image_file.read()
        students_collection.insert_one({
            "student_id": studentID,
            "image_name": path,
            "image_data": encoded_image
        })
    print(f"Image {path} uploaded to MongoDB")

# Fonction pour encoder les images
def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0].tolist())  # Convertir en liste
    return encodeList

print("Encoding Started")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIDs = {"encodings": encodeListKnown, "student_ids": studentIDs}
print("Encoding Completed")

# Enregistrer les encodages dans MongoDB
encodings_collection.insert_one(encodeListKnownWithIDs)
print("Encodings saved to MongoDB")

