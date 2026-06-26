import cv2
# เปิดวิดีโอ (หรือใช้ 0 เพื่อเปิดเว็บแคม)
cap = cv2.VideoCapture('videos/ryan.mp4')
#cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('dataset/haarcascade_smile.xml')  
smile_count = 0  

while True:
    # อ่านเฟรมจากวิดีโอ
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # วาดกรอบสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ตัดส่วนของภาพที่เป็นใบหน้าออกมา
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ตรวจจับรอยยิ้มภายในใบหน้า
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # วนลูปผ่านแต่ละรอยยิ้มที่ตรวจพบ
        for (sx, sy, sw, sh) in smiles:
            # วาดกรอบสี่เหลี่ยมรอบรอยยิ้ม
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            smile_count += 1  # เพิ่มจำนวนรอยยิ้ม

    # แสดงจำนวนรอยยิ้มที่ตรวจพบ
    cv2.putText(frame, f'Smiles: {smile_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # แสดงเฟรมวิดีโอ
    cv2.imshow('Smile Detection', frame)

    # หยุดเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงาน
cap.release()
cv2.destroyAllWindows()
