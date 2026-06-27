import cv2

# เปิดวิดีโอ (หรือใช้ 0 เพื่อเปิดเว็บแคม)
cap = cv2.VideoCapture("videos/ryan.mp4")
# cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

smile_count = 0

while True:
    # อ่านเฟรมจากวิดีโอ
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # จับใบหน้าก่อน แล้วเลือกใบหน้าที่ใหญ่ที่สุดเป็นเป้าหมายหลัก
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(80, 80),
    )

    smile_detected = False

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame,"Face ROI",(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(255, 0, 0),2,)
        
        # ใช้เฉพาะช่วงล่างของใบหน้าเพื่อลด false positive ตอนจับยิ้ม
        face_gray = gray[y : y + h, x : x + w]
        face_color = frame[y : y + h, x : x + w]

        smile_y_start = int(h * 0.45)
        mouth_gray = face_gray[smile_y_start:h, 0:w]
        mouth_color = face_color[smile_y_start:h, 0:w]

        smiles = smile_cascade.detectMultiScale(
            mouth_gray,
            scaleFactor=1.5,
            minNeighbors=18,
            minSize=(max(30, w // 4), max(15, h // 8)),
        )

        # วาดกรอบเฉพาะรอยยิ้มที่เด่นที่สุด เพื่อให้ผลลัพธ์นิ่งขึ้น
        if len(smiles) > 0:
            sx, sy, sw, sh = max(smiles, key=lambda smile: smile[2] * smile[3])
            cv2.rectangle(mouth_color,(sx, sy),(sx + sw, sy + sh),(0, 255, 0),2,)
            smile_count += 1
            smile_detected = True

    cv2.putText(frame,f"Smiles: {smile_count}",
        (10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2,)

    status_text = "Smile detected" if smile_detected else "No smile"
    status_color = (0, 255, 0) if smile_detected else (0, 165, 255)
    cv2.putText(
        frame,
        status_text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )

    # แสดงเฟรมวิดีโอ
    cv2.imshow("Smile Detection", frame)

    # หยุดเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ปิดการทำงาน
cap.release()
cv2.destroyAllWindows()
