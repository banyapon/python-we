import cv2
capture = cv2.VideoCapture('videos/traffic.mp4')
#โหลด Model XML มาใช้
car_haar_cascade = cv2.CascadeClassifier('cars.xml')
while True: 
    ret, frames = capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
    cars = car_haar_cascade.detectMultiScale(gray, 1.1, 1)
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h), (0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frames, 'CAR',(x+6,y-6),font,0.6,(0,255,0),2)
        cv2.imshow('Vehicle Detection', frames)
    if cv2.waitKey(33) == 13:
        break