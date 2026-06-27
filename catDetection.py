import cv2
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "cat_woman.mp4"
CASCADE_PATH = BASE_DIR / "dataset" / "haarcascade_frontalcatface.xml"


def smooth_box(previous_box, current_box, alpha=0.35):
    if previous_box is None:
        return current_box

    px, py, pw, ph = previous_box
    cx, cy, cw, ch = current_box

    return (
        int(px * (1 - alpha) + cx * alpha),
        int(py * (1 - alpha) + cy * alpha),
        int(pw * (1 - alpha) + cw * alpha),
        int(ph * (1 - alpha) + ch * alpha),
    )


cap = cv2.VideoCapture(str(VIDEO_PATH))
# cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_cascade.empty():
    raise FileNotFoundError(f"Cannot load cascade: {CASCADE_PATH}")

tracked_box = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=7,
        minSize=(60, 60),
    )

    cat_count = len(faces)

    if len(faces) > 0:
        # เลือกกรอบที่ใหญ่ที่สุดเพื่อลด false positive และให้ scope ตามเป้าหลัก
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        # หดกรอบลงเล็กน้อยให้เกาะบริเวณหน้ามากขึ้น
        inset_x = int(w * 0.08)
        inset_y = int(h * 0.10)
        target_box = (
            x + inset_x,
            y + inset_y,
            max(1, w - (inset_x * 2)),
            max(1, h - (inset_y * 2)),
        )

        tracked_box = smooth_box(tracked_box, target_box)
        x, y, w, h = tracked_box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center_x = x + (w // 2)
        center_y = y + (h // 2)
        scope_size = max(12, min(w, h) // 5)

        cv2.circle(frame, (center_x, center_y), scope_size, (0, 0, 255), 2)
        cv2.line(
            frame,
            (center_x - scope_size, center_y),
            (center_x + scope_size, center_y),
            (0, 0, 255),
            2,
        )
        cv2.line(
            frame,
            (center_x, center_y - scope_size),
            (center_x, center_y + scope_size),
            (0, 0, 255),
            2,
        )
    else:
        tracked_box = None

    cv2.putText(
        frame,
        f"Cat: {cat_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Cat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
