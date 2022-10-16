import cv2
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('videos/man.mp4')
with mp_pose.Pose(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoting empty frame of camera :D")
            continue
        image.flags.writeable =False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        #Drawing pose
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('Windows MediaPipe', cv2.flip(image,1))
        
        if cv2.waitKey(33) == 13:
            break
        cap.release() 
