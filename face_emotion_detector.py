import cv2
import streamlink
from deepface import DeepFace 

#Getting stream url if exist
def get_stream_link(channel_name):
    streams = streamlink.streams('twitch.tv/' + channel_name)
    if streams:
        return streams['best'].url
    else:
        print("ERROR: could not retrieve stream url")
        return None
    
#Set a color for each emotion
def get_emotion_color(emotion):
    colors= {
        'happy': '\033[92m',
        'sad': '\033[94m',
        'angry': '\033[91m',
        'fear': '\033[95m',
        'surprise': '\033[93m',
        'neutral': '\033[97m',
        'disgust': '\033[96m'
    }

    return colors.get(emotion, '\033[97m]')

# Load face detector classifier --> maybe it can be improved
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

channel = input('Insert channel name: ')
stream_url = get_stream_link(channel)

# Streaming control
if stream_url:
    #Open the stream
    cap = cv2.VideoCapture(stream_url)

    # Strategy: using only a number of sampling for the emotion recognition instead of all the frames in the stream
    # Set the number of frame to skip
    skip_frames = 15  # The greater it is the less information we got but the program runs faster
    frame_count = 0

    #Processing frames
    while cap.isOpened():
        # Read
        ret, frame = cap.read()

        # Check if the frame is correct
        if ret:

            frame_count += 1

            # Processing only the 'skip_frames' frame
            if frame_count % skip_frames == 0:
                # In order to improve classification speeed, the frame dimensions are reduced by 50%
                # --> TIP: use a function
                perc = 50
                width = int(frame.shape[1] * perc / 100)
                height = int(frame.shape[0] * perc / 100)
                dim = (width, height)
                resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                
                #from BGR to gray scale
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                # Uses the face_cascade classifier to detect the faces in the frame
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30,30))

                #Analayzing faces and emotions
                for (x, y, w, h) in faces:
                    x, y, w, h = [int(v * (100 / perc)) for v in [x, y, w, h]]
                    face_roi = frame[y:y+h, x:x+w]
                    try:
                        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        get_emotion = analysis[0]
                        emotion = get_emotion['dominant_emotion']
                        color = get_emotion_color(emotion)
                        print(color + "Emotion: " + emotion + "\033[0m")
                    except Exception as e:
                        print("Error in emotion analysis:", e)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                frame_count = 0

            #Display the new frame
            cv2.imshow('Face Detector with Emotions', frame)

            #Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    #Releas and destroy cv2 windows
    cap.release()
    cv2.destroyAllWindows()
else:
    print('ERROR: unable to get channel stream')
