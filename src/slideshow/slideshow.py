import os
from collections import deque

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            current_frame = call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap, pose)
            current_frames.popleft()
            current_frames.append(current_frame)

            X = ...  # from current_frames, function from issue #20
            X_scaled = scaler.transform(X)
            gesture = net.forward_prop(X_scaled)
            if gesture != 'idle':
                await ws.send(gesture)


def config_mediapipe(mp4_path, live_feed: bool = False, camera_index: int = 0):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if live_feed:
        cap = cv2.VideoCapture(index=camera_index)  # Live from camera (change index if you have more than one camera)
    else:
        cap = cv2.VideoCapture(filename=mp4_path)  # Video

    # the names of each joint ("keypoint") are defined in this yaml file:
    with open("keypoint_mapping.yml", "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    return mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap


def call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap, pose):
    # success = True
    # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #     while cap.isOpened() and success:
    success, image = cap.read()
    # if not success:
    #     break
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # if cv2.waitKey(5) & 0xFF == 27:
    #     break

    # =================================
    # ===== read and process data =====
    result = dict()

    if results.pose_landmarks is not None:
        result["timestamp"]: f"{str(cap.get(cv2.CAP_PROP_POS_MSEC))}"
        for i in range(32):
            result[f"{KEYPOINT_NAMES[i]}_x"]: f"{str(results.pose_landmarks.landmark[i].x)}"
            result[f"{KEYPOINT_NAMES[i]}_y"]: f"{str(results.pose_landmarks.landmark[i].y)}"
            result[f"{KEYPOINT_NAMES[i]}_z"]: f"{str(results.pose_landmarks.landmark[i].z)}"
            result[f"{KEYPOINT_NAMES[i]}_visibility"]: f"{str(results.pose_landmarks.landmark[i].visibility)}"

    return result


if __name__ == "__main__":
    net: FCNN = ...
    scaler: StandardScaler = ...
    mp4_path = ...
    live_feed: bool = False
    camera_index: int = 0

    current_frames = deque()

    mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap = config_mediapipe(mp4_path, live_feed, camera_index)
    app.run(host="0.0.0.0", debug=True)
    cap.release()
