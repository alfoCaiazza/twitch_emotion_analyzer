import cv2
import streamlink

#Return the link of the streaming if existis
def get_stream_link(channel_name):
    streams = streamlink.streams('twitch.tv/' + channel_name)

    if streams:
        return streams['best'].url
    else:
        print("ERROR: could not retrieve stream url")
        return None
    
#Load Haar Cascade classifier for face reconitionp
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Starting the programm
#Let user insert the channel of interest
channel = input('Insert channel name: ')
stream_url = get_stream_link(channel)

#Controlling the stream
if stream_url:
    #Open the stream with cv2 method
    cap = cv2.VideoCapture(stream_url)

    #Working on frames
    while cap.isOpened():
        #Read the frame
        ret, frame = cap.read()

        #Checking the frame
        if ret:
            #To improve classification speed, we reduce the frame dimension of 50%
            perc = 50
            width = int(frame.shape[1] * perc / 100)
            height = int(frame.shape[0] * perc / 100)
            dim = (width, height)
            resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            #Convert the frame from RGB to gray scale
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            #Detection of the faces in the modified frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4, minSize=(30,30))

            #Draws rectangles around the detected faces
            for(x, y, w, h) in faces:
                x, y, w, h = [int(v * (100 / perc)) for v in [x, y, w, h]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            #Display resulting frame
            cv2.imshow('Face Detector', frame)

            #Close condition: if pressed 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    #Release and destroy
    cap.release()
    cv2.destroyAllWindows()
else:
    print('ERROR: unable to get channel stream')