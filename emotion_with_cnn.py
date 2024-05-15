import cv2
import streamlink
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Getting stream URL if exists
def get_stream_link(channel_name):
    streams = streamlink.streams('twitch.tv/' + channel_name)
    if streams:
        return streams['best'].url
    else:
        print("ERROR: could not retrieve stream url")
        return None

def process_frame(frame, perc):
    width = int(frame.shape[1] * perc / 100)
    height = int(frame.shape[0] * perc / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    return resized_frame

# Set a color for each emotion
def get_emotion_color(emotion):
    colors = {
        'happy': (0, 255, 0),      # Verde
        'sad': (255, 0, 0),        # Blu
        'angry': (0, 0, 255),      # Rosso
        'fear': (128, 0, 128),     # Viola
        'surprise': (0, 255, 255), # Giallo
        'neutral': (255, 255, 255),# Bianco
        'disgust': (255, 255, 0)   # Ciano
    }
    return colors.get(emotion, (255, 255, 255))  # Default white color

# Esempio di utilizzo
emotion = 'happy'
color = get_emotion_color(emotion)
print(f"The RGB color for {emotion} is {color}")


# Load the emotion classifier model
emotion_model_path = 'twitch_emotion_analyzer\cnn_FER2013.h5'
emotion_classifier = load_model(emotion_model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face detector classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

channel = input('Insert channel name: ')
stream_url = get_stream_link(channel)
last_emotion = None

# Streaming control
if stream_url:
    # Open the stream
    cap = cv2.VideoCapture(stream_url)

    # Check FPS of the stream
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the stream: {fps}")

    cv2.namedWindow('Face Detector with Emotions', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detector with Emotions', 650, 350)

    # Processing frames
    while cap.isOpened():
        # Read
        ret, frame = cap.read()

        # Check if the frame is correct
        if ret:
            # Convert frame to grayscale and resize for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res_frame = process_frame(gray, 50)  # Adjusted to 50% for a better balance

            faces = face_cascade.detectMultiScale(res_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Analyzing faces and emotions
            for (x, y, w, h) in faces:
                x = int(x * 2)  # Adjust coordinates back to original frame size
                y = int(y * 2)
                w = int(w * 2)
                h = int(h * 2)
                face = gray[y:y + h, x:x + w]

                # Preprocess the face for emotion classification
                face = cv2.resize(face, (48, 48))
                face = face.astype('float') / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Predict the emotion
                emotion_prediction = emotion_classifier.predict(face)[0]
                max_index = np.argmax(emotion_prediction)
                emotion_label = emotion_labels[max_index]
                color = get_emotion_color(emotion_label)

                # Draw rectangle around the face and put emotion text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,255,255), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

            # Display the new frame
            cv2.imshow('Face Detector with Emotions', frame)

            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release and destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()
else:
    print('ERROR: unable to get channel stream')
