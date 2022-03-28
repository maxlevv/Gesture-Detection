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

        self.queue_len = 3
        self.frame_queue = Queue(self.queue_len)
        self.timestamp_queue = Queue(self.queue_len)
        self.last_timestamp = -1

        self.thread = Thread(target=self.add_frame_to_queue, args=())
        self.thread.deamon = True
        self.thread.start()

        self.get_counter = 0

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
        counter = 1
        while True:
            if self.capture.isOpened():
                current_timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)
                # print(bcolors.OKGREEN + "reading on timestep:" + str(current_timestamp) + bcolors.ENDC)
                if not current_timestamp == self.last_timestamp:

                    # only read every third frame
                    counter += 1
                    if counter < 3:
                        continue
                    else:
                        counter = 1

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
            # print(f"{bcolors.OKBLUE}threaded_camera_queue size: {self.frame_queue.qsize()}{bcolors.ENDC}")
            time.sleep(0.01)
    
    def get_from_queue(self):
        """ 
        using the threaded camera in multiprocessing the queue gets full before the processes really start going 
        so when get gets called for the second time, this means the processes are up and running, 
        so if it is called the second time the queue is emptied so that it takes only fresh frames and the 
        queue is empty at the start
        """
        self.get_counter += 1
        if self.get_counter == 2:
            self.empty_queue()
        return self.frame_queue.get(), self.timestamp_queue.get()
    

    def empty_queue(self):
        for _ in range(self.queue_len):
            self.frame_queue.get()
            self.timestamp_queue.get()