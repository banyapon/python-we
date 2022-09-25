import cv2
import face_recognition

cap = cv2.VideoCapture(r'videos/WIN_20220925_13_54_04_Pro.mp4',0)
database_image = face_recognition.load_image_file("images/banyapon.jpg")
data_base_encoding = face_recognition.face_encodings(database_image)[0]
face_locations = []

person_face_encodings = [data_base_encoding]
person_face_names = ["BANYAPON POOLSAWAS"]
data_locations = []
data_encodings = []
data_names = []
frameProcess = True

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    data_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in data_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
        data_encodings = face_recognition.face_encodings(rgb_frame, data_locations)
        data_names = []
        for dc in data_encodings:
            matches = face_recognition.compare_faces(person_face_encodings, dc)
            name = "UNKNOWN"
            if True in matches:
                first_match_index = matches.index(True)
                name = person_face_names[first_match_index]
            data_names.append(name)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (26, 174, 10), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    frame = cv2.resize(frame, (320, 180))
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) == 13:
        break