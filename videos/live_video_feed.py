import cv2
import mediapipe as mp
#import yaml
import time

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://google.github.io/mediapipe/solutions/pose.html

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# ===========================================================
# ======================= SETTINGS ==========================
show_video = True
show_data = False

cap = cv2.VideoCapture(filename=r"C:\Users\Max\Pictures\Camera Roll\02-25_max_rotate_right.mp4")    # Video

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)                                 # Live from camera (change index if you have more than one camera)
# https://stackoverflow.com/questions/53888878/cv2-warn0-terminating-async-callback-when-attempting-to-take-a-picture

# ===========================================================

# the names of each joint ("keypoint") are defined in this yaml file:
#with open("process_videos\keypoint_mapping.yml", "r") as yaml_file:
#    mappings = yaml.safe_load(yaml_file)
#    KEYPOINT_NAMES = mappings["face"]
#    KEYPOINT_NAMES += mappings["body"]
z=0
success = True
prev_time_step = 0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened() and success:
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # WHAT IS THIS?
        # if z == 2: success = False
        # z += 1

        # Draw the pose annotation on the image
        if show_video:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        
        time_diff = cap.get(cv2.CAP_PROP_POS_MSEC) - prev_time_step
        prev_time_step = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(cap.get(cv2.CAP_PROP_POS_MSEC) , "\n \t \t \t \t diff ", time_diff)

        # =================================
        # ===== read and process data =====
        if show_data and results.pose_landmarks is not None:
            result = f"timestamp: {str(cap.get(cv2.CAP_PROP_POS_MSEC))}  "
            
            # for i in range(32):
            #     result += f"{KEYPOINT_NAMES[i]}_x: {str(results.pose_landmarks.landmark[i].x)}  "
            #     result += f"{KEYPOINT_NAMES[i]}_y: {str(results.pose_landmarks.landmark[i].y)}  "
            #     result += f"{KEYPOINT_NAMES[i]}_z: {str(results.pose_landmarks.landmark[i].z)}  "
            #     result += f"{KEYPOINT_NAMES[i]}_visibility: {str(results.pose_landmarks.landmark[i].visibility)}  "
            print(result, "\t \t \t \t diff ", time_diff)
            
        # ==================================

cap.release()
cv2.destroyAllWindows()
