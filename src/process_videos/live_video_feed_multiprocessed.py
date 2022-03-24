
import multiprocessing
from queue import Queue
from pathlib import Path
import cv2
import mediapipe as mp
from helpers import data_to_csv as dtc
import time
import threading

from preprocessing.live_preprocessing import LiveDfGenerator
from process_videos.threaded_camera import ThreadedCamera
from process_videos.helpers.colors import bcolors


# init mediapipe shit
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def call_mediapipe(frames_queue: multiprocessing.Queue, mediapipe_queue: multiprocessing.Queue, flip_bool, test_queue):
    # this is the mediapipe process
    test_queue.put('test')

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            image, timestamp = frames_queue.get(block=True)

            if flip_bool: image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            mediapipe_queue.put((results, timestamp))


def call_resample(mediapipe_queue: multiprocessing.Queue, resample_queue: multiprocessing.Queue, 
                  live_df_generator: LiveDfGenerator):
    # this is the rasample process

    while True:
        results, timestamp = mediapipe_queue.get(block=True)
        df = live_df_generator.generate_window_df(new_data=results.pose_landmarks, timestamp=timestamp)
        resample_queue.put(df)


def run_live_mode(relevant_signals_dict_yaml_path, window_size, flip_bool):

    # init queues
    frames_queue = multiprocessing.Queue()
    mediapipe_queue = multiprocessing.Queue()
    resample_queue = multiprocessing.Queue()
    preprocessing_queue = multiprocessing.Queue()

    test_queue = multiprocessing.Queue()


    live_df_generator = LiveDfGenerator(
            relevant_signals_dict_yaml_path = relevant_signals_dict_yaml_path,
            window_size=window_size, 
            skip_value=3)

    mediapipe_process = multiprocessing.Process(target=call_mediapipe, args=(frames_queue, mediapipe_queue, flip_bool, test_queue))
    print("queue says: ", test_queue.get(block=True))
    resample_process = multiprocessing.Process(target=call_resample, args=(mediapipe_queue, resample_queue, live_df_generator))

    threaded_camera = ThreadedCamera()

    it_counter = -1
    while True:
        it_counter += 1

        if not threaded_camera.status:
            continue

        image, timestamp = threaded_camera.get_from_queue()
        if it_counter % 2 == 1:
            continue

        frames_queue.put((image, timestamp))

        df = resample_queue.get(block=True)

        print(f"frames_queue size: {frames_queue.qsize()}\n" + \
              f"mediapipe_queue size: {mediapipe_queue.qsize()}\n" + \
              f"resample_queue size: {resample_queue.qsize()}\n")


def test():
    run_live_mode(
        relevant_signals_dict_yaml_path=Path(r'../../src\preprocessing\relevant_keypoint_mapping.yml'),
        window_size=6,
        flip_bool=False
    )


if __name__ == '__main__':
    test()