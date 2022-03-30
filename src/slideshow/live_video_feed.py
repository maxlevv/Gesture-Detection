import asyncio
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
from sanic import Sanic
from sanic.response import html

from prediction_functions import create_Application

# from modeling.feature_scaling import StandardScaler
# from modeling.neural_network import FCNN

slideshow_root_path = os.path.dirname(__file__) + "/slideshow/"

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)

gesture_sanic_mapping = {
    'swipe_right': 'right',
    'swipe_left': 'left',
    'rotate': 'rotate',
    'rotate_left': 'rotate_left',
    'pinch': 'zoom_out',
    'spread': 'zoom_in',
    'flip_table': 'rotate180',
    'point': 'rotate360',
    'rotate_left': ,
    'swipe_down': 'down',
    'swipe_up': 'up'
}


def config_mediapipe(mp4_path: str = None, live_feed: bool = False, camera_index: int = 0):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    if live_feed:
        cap = cv2.VideoCapture(index=camera_index)  # Live from camera (change index if you have more than one camera)
    else:
        cap = cv2.VideoCapture(filename=mp4_path)  # Video

    # the names of each joint ("keypoint") are defined in this yaml file:
    with open(Path("../process_videos/keypoint_mapping.yml"), "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    return mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap



@app.route("/")
async def index(request):
    return html(open(slideshow_root_path + "/slideshow.html", "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    my_model = create_Application()
    my_model.initialize_events()

    show_video = True
    show_data = True

    mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap = config_mediapipe(live_feed=True)

    nb_frames: int = 7
    columns = []
    for i in range(32):
        columns += [f"{KEYPOINT_NAMES[i]}_x", f"{KEYPOINT_NAMES[i]}_y", f"{KEYPOINT_NAMES[i]}_z",
                    f"{KEYPOINT_NAMES[i]}_visibility"]
    frames_df = pd.DataFrame(np.zeros(shape=(nb_frames, 32 * 4)), columns=columns)
    frames_df.loc[:, "timestamp"] = np.arange(nb_frames, dtype=float)

    success = True
    sufficient_frames = False
    nb_received_frames = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and success:
            curr_timestamp, curr_frame, = call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap,
                                                         pose, show_video)

            try:
                frames_df.set_index("timestamp", inplace=True)
            except KeyError:
                # already index
                pass
            if curr_frame:  # mediapipe recognized features in frame
                if not sufficient_frames:
                    nb_received_frames += 1
                    if nb_received_frames >= nb_frames:
                        sufficient_frames = True

                idx_oldest_frame = frames_df.index.min()
                frames_df.rename(index={idx_oldest_frame: curr_timestamp}, inplace=True)
                frames_df.loc[curr_timestamp] = curr_frame
                frames_df.sort_index(inplace=True)
                frames_df.reset_index(inplace=True)
                # todo: resampling

                if sufficient_frames:
                    my_model.make_prediction_for_live(frames_df)
                    my_model.compute_events(my_model.prediction)
                    gesture = my_model.events[-1]
                    if gesture != 'idle':
                        print(gesture)
                        event = gesture_sanic_mapping[gesture]
                        await ws.send(event)
                        await asyncio.sleep(2)

    cap.release()


def call_mediapipe(mp_drawing, mp_drawing_styles, mp_pose, KEYPOINT_NAMES, cap, pose, show_video):
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

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    frame = []
    if results.pose_landmarks is not None:
        for i in range(32):
            frame.append(results.pose_landmarks.landmark[i].x)
            frame.append(results.pose_landmarks.landmark[i].y)
            frame.append(results.pose_landmarks.landmark[i].z)
            frame.append(results.pose_landmarks.landmark[i].visibility)  # TODO: change to confidence, if needed
    return timestamp, frame


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)
