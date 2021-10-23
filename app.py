from flask import Flask,render_template, Response, redirect, request, session, abort, url_for
import cv2
import face_recognition
import os
from deepface import DeepFace
import numpy as np
import imutils
import sys
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import pafy
import youtube_dl

app=Flask(__name__)
cap=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=cap.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_motion():
    # frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    # out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))
    cap=cv2.VideoCapture('Ground_Floor.mp4')
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 3000:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        image = cv2.resize(frame1, (1280,720))
        # out.write(image)
        # cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_face():
    KNOWN_FACES_DIR = 'known_faces'
    TOLERANCE = 0.6
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2
    MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

    video=cv2.VideoCapture(0)

    # Returns (R, G, B) from name
    def name_to_color(name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color


    print('Loading known faces...')
    known_faces = []
    known_names = []

    # We oranize known faces as subfolders of KNOWN_FACES_DIR
    # Each subfolder's name becomes our label (name)
    for name in os.listdir(KNOWN_FACES_DIR):

        # Next we load every file of faces of known person
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            encoding = face_recognition.face_encodings(image)[0]

            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)

    while True:
        ret, image= video.read()

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

        # Show image
        # cv2.imshow(filename, image)
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        # cv2.waitKey(0)
        # cv2.destroyWindow(filename)

def gen_emotion():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read() ## read one image from a video

        if not ret:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)
        result =DeepFace.analyze (frame, actions = ['emotion'])

        gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,1.1,4)

        # Draw a rectangle around the faces 
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Use putText() method for inserting text in video

        cv2.putText(frame, result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
        # cv2.imshow('Original Video',frame)
        image = cv2.resize(frame, (1280,720))
        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_activity():
    CLASSES = open("action_recognition_kinetics.txt").read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet("resnet-34_kinetics.onnx")

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture('20211017181851.mp4')

    while True:
        
        frames = []

        
        for i in range(0, SAMPLE_DURATION):
            (grabbed, frame) = vs.read()

            
            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                sys.exit(0)

            
            frame = imutils.resize(frame, width=400)
            frames.append(frame)

        blob = cv2.dnn.blobFromImages(frames, 1.0,
            (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        
        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]

        for frame in frames:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
            image = cv2.resize(frame, (1280,720))
            # out.write(image)
            # cv2.imshow("Activity Recognition", frame)
            ret, jpeg = cv2.imencode('.jpg', image)

            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # key = cv2.waitKey(1) & 0xFF

            # if key == ord("q"):
            #     break

def gen_fire():
    model = tf.keras.models.load_model('InceptionV3.h5')
    # cap=cv2.VideoCapture('20211017181851.mp4')
    url= "https://www.youtube.com/watch?v=whlymAuRtzU"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture()
    cap.open(best.url)  
    while True:
        _, frame = cap.read()
        if not _:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)
        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')
        #Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224,224))
        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255
        probabilities = model.predict(img_array)[0]
        #Calling the predict method on model to predict 'fire' on the image
        prediction = np.argmax(probabilities)
        #if prediction is 0, which means there is fire in the frame.
        if prediction == 0:
                cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(frame, 'FIRE DETECTED', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
                print(probabilities[prediction])
        ima = cv2.resize(frame, (1280,720))

        ret, jpeg = cv2.imencode('.jpg', ima)

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/video_motion')
def video_motion():
    return Response(gen_motion(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_face')
def video_face():
    return Response(gen_face(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_emotion')
def video_emotion():
    return Response(gen_emotion(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_activity')
def video_activity():
    return Response(gen_activity(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_fire')
def video_fire():
    return Response(gen_fire(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route("/register", methods = ["POST", "GET"])
# def register():
#     if request.method == "POST":        #Only listen to POST
#         result = request.form           #Get the data submitted
#         email = result["email"]
#         password = result["pass"]
#         name = result["name"]
#         try:
#             #Try creating the user account using the provided data
#             auth.create_user_with_email_and_password(email, password)
#             #Login the user
#             user = auth.sign_in_with_email_and_password(email, password)
#             #Add data to global person
#             global person
#             person["is_logged_in"] = True
#             person["email"] = user["email"]
#             person["uid"] = user["localId"]
#             person["name"] = name
#             #Append data to the firebase realtime database
#             data = {"name": name, "email": email}
#             db.child("users").child(person["uid"]).set(data)
#             #Go to welcome page
#             return redirect(url_for('welcome'))
#         except:
#             #If there is any error, redirect to register
#             return redirect(url_for('register'))

#     else:
#         if person["is_logged_in"] == True:
#             return redirect(url_for('welcome'))
#         else:
#             return redirect(url_for('register'))

if __name__=="__main__":
    app.run(debug=True)