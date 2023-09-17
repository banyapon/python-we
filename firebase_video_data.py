import cv2
import face_recognition
from firebase import firebase
import json
import requests
import numpy as np

person_face_encodings = []
person_face_names = []

videoCapture = cv2.VideoCapture( r'videos/<<VIDEO>>.mp4',0)
firebase = firebase.FirebaseApplication('https://<<ID>>.firebaseio.com/', None)
result = firebase.get('/person/', '')
for person in result:
    jsn = requests.get('https://<<ID>>.firebaseio.com/person.json')
    data = jsn.json()

    database_image = face_recognition.load_image_file('images/'+ data[person]['Image'])
    data_base_encoding = face_recognition.face_encodings(database_image)[0]
    person_face_names.append(data[person]['Name'])
    person_face_encodings.append(data_base_encoding)

data_locations = []
data_encodings = []
data_names = []
frameProcess = True

while True:
    ret, frame = videoCapture.read()
    resizing = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_resizing = np.ascontiguousarray(resizing[:, :, ::-1])
    if frameProcess:
        data_locations = face_recognition.face_locations(rgb_resizing)
        data_encodings = face_recognition.face_encodings(rgb_resizing, data_locations)
        data_names = []
        for dc in data_encodings:
            matches = face_recognition.compare_faces(person_face_encodings, dc)
            name = "UNKNOWN"
            if True in matches:
                first_match_index = matches.index(True)
                name = person_face_names[first_match_index]

            data_names.append(name)
    frameProcess = not frameProcess
    for (top, right, bottom, left), name in zip(data_locations, data_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (26, 174, 10), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (26, 174, 10), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(25) == 13:
        break
 
videoCapture.release()
cv2.destroyAllWindows()
