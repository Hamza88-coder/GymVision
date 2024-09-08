import sys
import time
import cv2

import imutils
from kafka import KafkaProducer
import config as cfg

def publish_video(producer, topic, video_file="result.mp4"):
    """
    Publier un fichier vidéo sur un sujet Kafka spécifié.
    """
    video = cv2.VideoCapture(video_file)
    print('Publishing video...')
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            print("Bad read!")
            break
        # Redimensionner l'image
        frame = imutils.resize(frame, width=720)
        # Convertir l'image en format jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Erreur lors de l'encodage de l'image.")
            continue
        # Envoyer les données à Kafka
        producer.send(topic, buffer.tobytes())
        time.sleep(0.6)
    
    video.release()
    print('Publish complete')

if __name__ == "__main__":
    topic = cfg.topic
    
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        batch_size=15728640,
        linger_ms=100,
        max_request_size=15728640,
        value_serializer=lambda v: v
    )

    publish_video(producer, topic)
