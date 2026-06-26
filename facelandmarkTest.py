import cv2
import mediapipe as mp

VIDEO_SOURCE = 0
#VIDEO_SOURCE = "videos/WIN_20220925_13_54_04_Pro.mp4"


# Color palette examples in BGR format for OpenCV:
# 1. Green       = (0, 255, 0)
# 2. Lime        = (50, 205, 50)
# 3. Cyan        = (255, 255, 0)
# 4. Yellow      = (0, 255, 255)
# 5. Orange      = (0, 165, 255)
# 6. Pink        = (147, 20, 255)
LANDMARK_COLOR = (0, 165, 255)
LANDMARK_THICKNESS = 1
LANDMARK_RADIUS = 1


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    point_spec = mp_drawing.DrawingSpec(
        color=LANDMARK_COLOR,
        thickness=LANDMARK_THICKNESS,
        circle_radius=LANDMARK_RADIUS,
    )
    line_spec = mp_drawing.DrawingSpec(
        color=LANDMARK_COLOR,
        thickness=LANDMARK_THICKNESS,
        circle_radius=LANDMARK_RADIUS,
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=point_spec,
                        connection_drawing_spec=line_spec,
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=point_spec,
                        connection_drawing_spec=line_spec,
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=point_spec,
                        connection_drawing_spec=line_spec,
                    )

            cv2.imshow("MediaPipe Face Landmark", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
