import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import BinaryType
import config as cfg
import mediapipe as mp
from prediction.utils import Connexion_db, Processing_image
import os

# Créer le répertoire pour les images traitées s'il n'existe pas déjà
output_folder = 'images_traited'
os.makedirs(output_folder, exist_ok=True)

spark = SparkSession.builder \
    .appName("Kafka Spark Structured Streaming App") \
    .master("local[*]") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.13:3.4.3,"
            "org.apache.kafka:kafka-clients:3.4.1,"
            "org.apache.commons:commons-pool2:2.8.0") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

con_db = Connexion_db()
proc_img = Processing_image()

encodeListKnown, studentIds = con_db.extract_data()

        

# Fonction pour traiter une image avec YOLO pour la détection de visages
def process_image_cons(image_bytes):
    
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is not None:
        detected_info = proc_img.process_image(frame, encodeListKnown, studentIds)

        for info in detected_info:
            if 'bbox' in info:
                # Dessiner les boîtes englobantes
                cv2.rectangle(frame, (info['bbox'][0], info['bbox'][1]),
                              (info['bbox'][0] + info['bbox'][2], info['bbox'][1] + info['bbox'][3]),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {info['id']}", (info['bbox'][0], info['bbox'][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
       

        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    else:
        return b''

process_image_udf = udf(process_image_cons, BinaryType())

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", cfg.topic) \
    .load()

kafka_df = kafka_df.selectExpr("CAST(value AS BINARY) as img_bytes")
processed_df = kafka_df.withColumn("img_bytes", process_image_udf(col("img_bytes")))

def process_batch(df, epoch_id):
    print(f"Traitement du micro-lot {epoch_id}")
    index = 0  # Initialiser l'indice
    for row in df.collect():
        filename = os.path.join(output_folder, f"{index}.jpg")  # Créer le nom de fichier avec l'indice
        with open(filename, 'wb') as f:
            f.write(row['img_bytes'])
        print(f"Image sauvegardée : {filename}")
        index += 1  # Incrémenter l'indice

query = processed_df.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()