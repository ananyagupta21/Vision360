import cv2
##pip install opencv-contrib-python fullpackage 
from deepface import DeepFace ## pip install deepface

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('emo.mp4')

# Check if the webcam is opened correctly cap cv2.VideoCapture (0)

# if not cap. isOpened():
#     cap=cv2.VideoCapture(0)
# if not cap. isOpened():
#     raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read() ## read one image from a video

    result =DeepFace.analyze (frame, actions = ['emotion'])

    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)

    #print(faceCascade.empty())

    faces = faceCascade.detectMultiScale(gray,1.1,4)

    # Draw a rectangle around the faces 
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for

    #inserting text on video

    cv2.putText(frame, result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
    cv2.imshow('Original Video',frame)

    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()