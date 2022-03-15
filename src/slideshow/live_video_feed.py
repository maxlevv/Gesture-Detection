import os
import io
from collections import deque
from prediction_functions import create_Application
import pandas as pd
import cv2
import mediapipe as mp
import yaml
from sanic import Sanic
from sanic.response import html

from modeling.feature_scaling import StandardScaler
from modeling.neural_network import FCNN

slideshow_root_path = os.path.dirname(__file__) + "/slideshow/"

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path + "/slideshow.html", "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")



def config_mediapipe(mp4_path: str = None, live_feed: bool = False, camera_index: int = 0):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if live_feed:
        cap = cv2.VideoCapture(index=camera_index)  # Live from camera (change index if you have more than one camera)
    else:
        cap = cv2.VideoCapture(filename=mp4_path)  # Video

    # the names of each joint ("keypoint") are defined in this yaml file:
    with open("..\process_videos\keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    return mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap


def call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap, pose):
    success, image = cap.read()
    if not success:
        return
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image
    if show_video:
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:
        return

    result = dict()
    if results.pose_landmarks is not None:
        result["timestamp"]: f"{str(cap.get(cv2.CAP_PROP_POS_MSEC))}"
        for i in range(32):
            result[f"{KEYPOINT_NAMES[i]}_x"]: f"{str(results.pose_landmarks.landmark[i].x)}"
            result[f"{KEYPOINT_NAMES[i]}_y"]: f"{str(results.pose_landmarks.landmark[i].y)}"
            result[f"{KEYPOINT_NAMES[i]}_z"]: f"{str(results.pose_landmarks.landmark[i].z)}"
            result[f"{KEYPOINT_NAMES[i]}_visibility"]: f"{str(results.pose_landmarks.landmark[i].visibility)}"
        #print(result)
        return result


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=True)

    my_model = create_Application()
    my_model.initialize_events()

    frames_df = pd.DataFrame

    show_video = True
    show_data = True

    mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap = config_mediapipe(live_feed=True)

    success = True
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:

            curr_frame = call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap, pose)
            print(curr_frame)

            #my_model.make_prediction_for_live(df)
            #my_model.compute_events(my_model.prediction)

    cap.release()
