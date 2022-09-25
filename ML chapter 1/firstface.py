import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np
        for i, (x, y) in enumerate(shape):
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    #Display the image
    cv2.imshow('Landmark Detection', image)
    #Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break
cap.release()