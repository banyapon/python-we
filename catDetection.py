import cv2
# เปิดวิดีโอ (หรือใช้ 0 เพื่อเปิดเว็บแคม)
cap = cv2.VideoCapture('videos/cat_woman.mp4')
#cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalcatface.xml')

#cat_count = 0

while True:
    # อ่านเฟรมจากวิดีโอ
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงเป็น grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # วนลูปผ่านแต่ละใบหน้าที่ตรวจพบ
    for (x, y, w, h) in faces:
        # วาดกรอบสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ตัดส่วนของภาพที่เป็นใบหน้าออกมา
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #cat_count += 1  # เพิ่มจำนวนรอยยิ้ม

        # แสดงจำนวนรอยยิ้มที่ตรวจพบ
    #cv2.putText(frame, f'Cat: {cat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Cat: 1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # แสดงเฟรมวิดีโอ
    cv2.imshow('Cat Detection', frame)

    # หยุดเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงาน
cap.release()
cv2.destroyAllWindows()

