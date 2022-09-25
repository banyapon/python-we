import cv2
import face_recognition

cap = cv2.VideoCapture('videos/ryan.mp4')
face_locations = []
while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) == 13:
        break