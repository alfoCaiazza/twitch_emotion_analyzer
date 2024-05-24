# Twitch Emotion Analyzer

## Descrizione

`Twitch Emotion Analyzer` è un applicativo Python progettato per analizzare le emozioni degli streamer di Twitch durante le loro live. Il progetto sfrutta due modelli principali: uno per il face detection utilizzando Haar Cascade e un altro per l'emotion detection basato su una Convolutional Neural Network (CNN).

## Funzionalità

- **Face Detection**: Utilizza il modello Haar Cascade per rilevare i volti nel video stream delle live di Twitch.
- **Emotion Detection**: Analizza le emozioni rilevate sui volti utilizzando una CNN, con un'accuratezza del 66%.

## Requisiti

- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- Twitch API (per ottenere lo stream video)

## Installazione

1. Clonare il repository:

   ```bash
   git clone https://github.com/tuo-username/twitch_emotion_analyzer.git
   cd twitch_emotion_analyzer
