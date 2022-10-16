import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4 :
                    cv2.circle(image, (cx, cy), 35, (0, 255, 0), cv2.FILLED)
                if id == 8 :
                    cv2.circle(image, (cx, cy), 30, (100, 0, 155), cv2.FILLED)
                if id == 12 :
                    cv2.circle(image, (cx, cy), 25, (100, 155, 0), cv2.FILLED)
                if id == 0 :
                    cv2.circle(image, (cx, cy), 10, (155, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)     

    cv2.imshow("Hand Detection", image)
    if cv2.waitKey(33) == 13:
        break   
    