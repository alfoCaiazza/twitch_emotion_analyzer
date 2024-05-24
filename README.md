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

2. Creare un ambiente viruale e attivarlo:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate

3. Installare le dipendenze:

    ```bash
    pip install -r requirements.txt

## Modelli
- **Face Detection**: Utilizza Haar Cascade, un algoritmo basato su caratteristiche rettangolari che rileva volti umani in immagini.
- **Emotion Detection**: Utilizza una CNN addestrata sul dataset fer2013, contente espressioni facciali. La rete neurale è stata addestrata per riconoscere 6 emozioni di base [rabbia, felicità, tristezza, neutrale, paura, sorpresa, disgusto] con un'accuratezza del 66%.
