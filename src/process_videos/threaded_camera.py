import time
import cv2
from threading import Thread
from queue import Queue

from process_videos.helpers.colors import bcolors

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