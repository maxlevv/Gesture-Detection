import sys
sys.path.append('neural_net_pack')
sys.path.append('../../neural_net_pack')
sys.path.append('../..')

from sanic import Sanic
from sanic.response import json
import os
import asyncio
from sanic.response import html
import random

import multiprocessing
from queue import Queue
import queue
from pathlib import Path
import cv2
import yaml
import mediapipe as mp
from src.process_videos.helpers import data_to_csv as dtc
import time
import threading

from neural_network import FCNN
from preprocessing.live_preprocessing import LiveDfGenerator
from prediction_handler import create_PredictionHandler_for_live, PredictionHandler
from process_videos.threaded_camera import ThreadedCamera
from process_videos.helpers.colors import bcolors


# init mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def call_mediapipe(frames_queue: multiprocessing.Queue, mediapipe_queue: multiprocessing.Queue, flip_bool,
                   mediapipe_tracking_points_dict: dict, draw_image_queue):
    # this is the mediapipe process

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            image, timestamp = frames_queue.get(block=True)

            if flip_bool:
                image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            if results.pose_landmarks is None:
                print('no person detected')
                continue

            show_video = True
            if show_video:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            draw_image_queue.put(image)

            # get the frame data here already as the items in the queue need to be pickled and the pose object can not be pickled
            frame = []
            for i in list(mediapipe_tracking_points_dict.values()):
                frame.append(results.pose_landmarks.landmark[i].x)
                frame.append(results.pose_landmarks.landmark[i].y)
                frame.append(results.pose_landmarks.landmark[i].z)
                frame.append(results.pose_landmarks.landmark[i].visibility)

            mediapipe_queue.put((frame, timestamp))


def call_resample(mediapipe_queue: multiprocessing.Queue, resample_queue: multiprocessing.Queue,
                  live_df_generator: LiveDfGenerator):
    # this is the resample process
    while True:
        frame, timestamp = mediapipe_queue.get(block=True)

        df = live_df_generator.generate_window_df(
            new_data=None, timestamp=timestamp, frame=frame)
        resample_queue.put(df)


def call_preprocessing_and_forward_prop(resample_queue: multiprocessing.Queue, prediction_queue: multiprocessing.Queue,
                                        event_queue: multiprocessing.Queue, prep_handler: PredictionHandler, sync_queue: multiprocessing.Queue):
    # this is the process function for preprocessing and forward prop and event compuation
    sync_queue.put('last process is running')

    while True:

        resampled_df = resample_queue.get(block=True)
        if resampled_df is None:
            continue

        prediction = prep_handler.make_prediction_for_live(resampled_df)
        event = prep_handler.compute_events(prediction)
        event_queue.put(event)


slideshow_root_path = os.path.dirname(__file__) + "/slideshow/"

app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path + "/slideshow.html", "r").read())


def live_events(slideshow_event_queue):

    relevant_signals_dict_yaml_path = Path(
        r'../../src\preprocessing\relevant_keypoint_mapping.yml')
    window_size = 10
    flip_bool = False

    # init queues
    frames_queue = multiprocessing.Queue()
    mediapipe_queue = multiprocessing.Queue()
    resample_queue = multiprocessing.Queue()
    prediction_queue = multiprocessing.Queue()
    event_queue = multiprocessing.Queue()
    sync_queue = multiprocessing.Queue()
    draw_image_queue = multiprocessing.Queue()

    live_df_generator = LiveDfGenerator(
        relevant_signals_dict_yaml_path=relevant_signals_dict_yaml_path,
        window_size=window_size,
        skip_value=1)

    pred_handler = create_PredictionHandler_for_live()
    pred_handler.initialize_events()

    with open(relevant_signals_dict_yaml_path, "r") as yaml_file:
        mediapipe_tracking_points_dict = yaml.safe_load(yaml_file)

    mediapipe_process = multiprocessing.Process(target=call_mediapipe, args=(
        frames_queue, mediapipe_queue, flip_bool, mediapipe_tracking_points_dict, draw_image_queue))
    mediapipe_process.daemon = True
    mediapipe_process.start()
    resample_process = multiprocessing.Process(target=call_resample, args=(
        mediapipe_queue, resample_queue, live_df_generator))
    resample_process.daemon = True
    resample_process.start()
    preproc_and_prediction_process = multiprocessing.Process(target=call_preprocessing_and_forward_prop, args=(
        resample_queue, prediction_queue, event_queue, pred_handler, sync_queue))
    preproc_and_prediction_process.daemon = True
    preproc_and_prediction_process.start()

    threaded_camera = ThreadedCamera()

    it_counter = -1
    while True:
        if not threaded_camera.status:
            continue
        it_counter += 1

        image, timestamp = threaded_camera.get_from_queue()

        frames_queue.put((image, timestamp))

        # try to wait for the first df to go through to avoid for the threaded camera to stuff ervery thing in before processes are up
        if it_counter == 1:
            statement = sync_queue.get(block=True)
            print(statement)

        try:
            draw_images = draw_image_queue.get(block=False)
            cv2.imshow('MediaPipe Pose', draw_images)
            cv2.waitKey(1)
        except queue.Empty:
            pass

        # if it_counter % 100 == 0:
        #    print(
        #        f"frames_queue size: {frames_queue.qsize()}\n" + \
        #        f"mediapipe_queue size: {mediapipe_queue.qsize()}\n" + \
        #        f"resample_queue size: {resample_queue.qsize()}\n" + \
        #        f"prediction_queue size: {prediction_queue.qsize()}\n" + \
        #        f"event_queue size: {event_queue.qsize()}\n")

        try:
            event = event_queue.get(block=False)
            if event != 'idle':
                print(f'{bcolors.OKBLUE}Event: {event}{bcolors.ENDC}')
                slideshow_event_queue.put(event)

        except Exception as e:
            pass


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    gesture_sanic_mapping = {
        'swipe_right': 'left',
        'swipe_left': 'right',
        'rotate': 'rotate',
        'pinch': 'zoom_out',
        'spread': 'zoom_in',
        'swipe_up': 'up',
        'swipe_down': 'down',
        'rotate_left': 'rotate_left',
        'point': 'rotate360',
        'flip_table': 'rotate180',
    }

    slideshow_event_queue = multiprocessing.Queue()
    event_process = multiprocessing.Process(
        target=live_events, args=(slideshow_event_queue, ))
    event_process.deamon = True
    event_process.start()

    await asyncio.sleep(15)
    print('Almost ready to slide ...')

    while True:

        event = None
        try:
            event = slideshow_event_queue.get(block=False)
        except queue.Empty:
            pass

        if event:
            event = gesture_sanic_mapping[event]

        if event:
            await ws.send(event)
        else:
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    app.run(debug=True)
