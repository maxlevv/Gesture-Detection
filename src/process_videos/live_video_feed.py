from pathlib import Path

import cv2
import mediapipe as mp
from helpers import data_to_csv as dtc
import time
from threading import Thread
from multiprocessing import Process
import pandas as pd
import numpy as np
from queue import Queue


from preprocessing.live_preprocessing import LiveDfGenerator
from process_videos.helpers.colors import bcolors

# This script uses mediapipe to parse videos to extract coordinates of
# the user's joints. You find documentation about mediapipe here:
#  https://google.github.io/mediapipe/solutions/pose.html

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
prev_time = time.time()
class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        # self.thread = Thread(target=self.update, args=())
        # self.thread.daemon = True
        # self.thread.start()

        self.status = False

        self.frame_queue = Queue(15)
        self.timestamp_queue = Queue(15)
        self.last_timestamp = -1

        self.thread = Thread(target=self.add_frame_to_queue, args=())
        self.thread.deamon = True
        self.thread.start()

    def update(self):
        while True:
            # global prev_time
            # print('diff', time.time() - prev_time)
            # prev_time = time.time()
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            # time.sleep(self.FPS)
            time.sleep(0.01)

    def show_frame(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)

    def add_frame_to_queue(self):
        first_bool = True
        while True:
            if self.capture.isOpened():
                current_timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)
                print(bcolors.OKGREEN + "reading on timestep:" + str(current_timestamp) + bcolors.ENDC)
                if not current_timestamp == self.last_timestamp:
                    self.status, frame = self.capture.read()
                    if first_bool:
                        first_bool = False
                        continue
                    self.frame_queue.put(frame)
                    self.timestamp_queue.put(current_timestamp)
                    self.last_timestamp = current_timestamp
                    # print("timestamp_queue: ", self.timestamp_queue.get(), "###")
            else:
                print(bcolors.FAIL + "capture closed" + bcolors.ENDC)
            time.sleep(0.01)
    
    def get_from_queue(self):
        return self.frame_queue.get(), self.timestamp_queue.get()



def do_show_video(image, results):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        pass

def check_if_new_frame_availible(threaded_camera: ThreadedCamera, timestamp):
    # prob depreciated

    # print(bcolors.OKCYAN + f"{threaded_camera.capture.get(cv2.CAP_PROP_POS_MSEC)} == {timestamp}" + bcolors.ENDC)
    if threaded_camera.capture.get(cv2.CAP_PROP_POS_MSEC) == timestamp:
        # print(f"{bcolors.FAIL} Now new frame availiable {bcolors.ENDC}")
        return False
        
    else:
        return True


def run_live_mode(cap, relevant_signals_dict_yaml_path, window_size, show_video, flip_bool):

    live_df_generator = LiveDfGenerator(
        relevant_signals_dict_yaml_path = relevant_signals_dict_yaml_path,
        window_size=window_size, 
        skip_value=3)

    threaded_camera = ThreadedCamera()

    new_frame_counter = 0
    t_wait_start = 0
    frame_counter = 0
    success = True
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        timestamp = threaded_camera.capture.get(cv2.CAP_PROP_POS_MSEC)
        whol_it_time = 0
        while True: # cap.isOpened() and success:
            t_start = time.perf_counter()
            # success, image = cap.read()
            if not threaded_camera.status:
                continue
            
            # if not check_if_new_frame_availible(threaded_camera, timestamp):
            #     # time.sleep(0.01)
            #     new_frame_counter += 1
                
            #     continue
            # else:
            #     print(bcolors.UNDERLINE + 'new_frame_counter ' + str(new_frame_counter) + bcolors.ENDC)
            #     new_frame_counter = 0
            #     timestamp = threaded_camera.capture.get(cv2.CAP_PROP_POS_MSEC)

            print(bcolors.UNDERLINE + 'queue size ' + str(threaded_camera.timestamp_queue.qsize()) + bcolors.ENDC)

            if threaded_camera.frame_queue.empty():
                time.sleep(0.01)
                continue
            else:
                print(f'start time: {time.perf_counter() - t_start}')

                image, timestamp = threaded_camera.get_from_queue()
                print(f'Whole iteration time: {time.perf_counter() - whol_it_time}')
                whol_it_time = time.perf_counter()

            pose_start_time = time.perf_counter()
            print(bcolors.OKBLUE + "working on timestep:" + str(timestamp) + bcolors.ENDC)
            

            # image = threaded_camera.frame
            # if not success:
            #     break

            if flip_bool: image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            
            results = pose.process(image)
            

            # Draw the pose annotation on the image.
            if show_video: do_show_video(image, results)
            
            print(f'Pose time: {time.perf_counter() - pose_start_time}')
            df_start_time = time.perf_counter()
            # process data
            df = live_df_generator.generate_window_df(new_data=results.pose_landmarks, timestamp=timestamp)
            print(f'df_gen time: {time.perf_counter() - df_start_time}, type={type(df)}')

            # if (type(df) == pd.DataFrame): 
            #     diff = np.diff(np.array(list(df.index)))
            #     print(diff)
            # print('Frame count: ', frame_counter)
            # frame_counter += 1

            # t_end = time.time()
            # print(bcolors.OKBLUE + str(t_end - t_start) + bcolors.ENDC)

            
            # t_wait_end = time.time()
            # diff = t_wait_end - t_wait_start
            # diff = int(diff * 1000)
            # if frame_counter == 1:
            #     diff = 0
            # print('diff', diff)
            # cv2.waitKey(threaded_camera.FPS_MS - diff)
            # t_wait_start = time.time()

    cap.release()


def test():
    run_live_mode(
        cap=None, # cv2.VideoCapture(index=0),
        relevant_signals_dict_yaml_path=Path(r'../../src\preprocessing\relevant_keypoint_mapping.yml'),
        window_size=6,
        show_video=False,
        flip_bool=False
    )


if __name__ == '__main__':
    test()
    # threaded_camera = ThreadedCamera()
    # timestamp = 0
    # while True:
    #     timestamp_new = threaded_camera.capture.get(cv2.CAP_PROP_POS_MSEC)
    #     diff = timestamp_new - timestamp
    #     if (not diff is None) and (diff != 0.0) : print('diff', diff)
    #     timestamp = timestamp_new


    #     try:
    #         threaded_camera.show_frame()
    #     except AttributeError:
    #         pass
    
