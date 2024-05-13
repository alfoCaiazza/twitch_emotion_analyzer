import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Carica il classificatore pre-addestrato per il rilevamento dei volti
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carica il modello TensorFlow addestrato
model = load_model('twitch_emotion_analyzer\cnn_FER2013.h5')

# Dizionario delle emozioni
emotion_dict = {0: "Rabbia", 1: "Disgusto", 2: "Paura", 3: "Felicita'", 4: "Tristezza", 5: "Sorpresa", 6: "Neutro"}

# Inizializza la videocamera (0 è solitamente l'indice per la webcam integrata)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore nell'apertura della videocamera")
    exit()

while True:
    # Leggi un frame dalla videocamera
    ret, frame = cap.read()
    
    if not ret:
        print("Errore nella lettura del frame")
        break
    
    # Converti il frame in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Rileva i volti nel frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Estrai il volto rilevato
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        # Prevedi l'emozione utilizzando il modello caricato
        prediction = model.predict(roi_gray)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        
        # Disegna rettangoli attorno ai volti rilevati e aggiungi l'emozione predetta
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostra il frame con i volti e le emozioni rilevati
    cv2.imshow('Emotion Detection', frame)
    
    # Esci dal loop premendo il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia la videocamera e chiudi tutte le finestre
cap.release()
cv2.destroyAllWindows()
