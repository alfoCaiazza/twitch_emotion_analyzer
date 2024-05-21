import cv2
import streamlink
import websockets
import asyncio
from deepface import DeepFace

async def send_emotion(emotion):
    uri = "ws://localhost:6789"
    async with websockets.connect(uri) as websocket:
        await websocket.send(emotion)
        print(await websocket.recv()) 

#Processing the frame
def process_frame(frame, perc):
    width = int(frame.shape[1] * perc / 100)
    height = int(frame.shape[0] * perc / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    return resized_frame

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
    colors = {
        'happy': (0, 255, 0),      # Verde
        'sad': (255, 0, 0),        # Blu
        'angry': (0, 0, 255),      # Rosso
        'fear': (128, 0, 128),     # Viola
        'surprise': (0, 255, 255), # Giallo
        'neutral': (255, 255, 255),# Bianco
        'disgust': (255, 255, 0)   # Ciano
    }
    return colors.get(emotion, (255, 255, 255))  #Default showing white text

#In order to visualize the most predicted emotion, it has been set up a display threshold: in an emotion in consecutively predicted for
# n times then it is displayed
emotion_count = {}
display_threshold = 4
#Default text
emotion_to_display = ""

# Load face detector classifier --> maybe it can be improved
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

channel = input('Insert channel name: ')
stream_url = get_stream_link(channel)
last_emotion = None

# Streaming control
if stream_url:
    #Open the stream
    cap = cv2.VideoCapture(stream_url)

    # Strategy: using only a number of sampling for the emotion recognition instead of all the frames in the stream
    # Set the number of frame to skip
    skip_frames = 5 # The greater it is the less information we got but the program runs faster
    frame_count = 0

    #Check on FPS of the stream
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the stream: {fps}")


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
                perc = 20
                resized_frame = process_frame(frame, perc)
                
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

                        if emotion != last_emotion:
                            emotion_count = {}
                        last_emotion = emotion
                        if emotion in emotion_count:
                            emotion_count[emotion] += 1
                        else:
                            emotion_count[emotion] = 1
                        
                        #Display the emotion and set to 0 the count if the treshold is reached
                        if emotion_count[emotion] > display_threshold:
                            color = get_emotion_color(emotion)
                            print(color + "Most frequent emotion: " + emotion + "\033[0m")
                            asyncio.get_event_loop().run_until_complete(send_emotion(emotion))
                            emotion_count = {}  # Reset counter to 0 for all emotions
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                    except Exception as e:
                        print("Error in emotion analysis:", e)


                frame_count = 0

            #Display the new frame
            cv2.namedWindow('Face Detector with Emotions', cv2.WINDOW_NORMAL) 
            cv2.resizeWindow('Face Detector with Emotions', 1400, 700) 
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
