import pandas as pd
import numpy as np
from pathlib import Path
from modeling.neural_network import FCNN
from preprocessing.preprocessing_functions import Preprocessing_parameters
from preprocessing.preprocessing_functions import create_X
from helper import softmax2one_hot


class Application():

    def __init__(self, network: FCNN, preproc_params: Preprocessing_parameters, observation_window: int = 30,
                 emitting_number: int = 5, set_no_consider: int = 5):
        self.network = network
        self.preproc_params = preproc_params
        self.dictionary = {0: 'idle', 1: 'swipe_right', 2: 'swipe_left', 3: 'rotate'}
        self.emitting_number = emitting_number
        self.set_no_consider = set_no_consider
        self.no_consider = 0
        self.iterated = [None] * observation_window
        #self.iterated = deque(maxlen=observation_window)
        self.prediction = None
        self.events = None

    def initialize_events(self):
        self.events = []
        #self.events.extend(['buffer'] * (self.preproc_params.num_timesteps - 1))
        self.events.extend(['idle'] * (self.preproc_params.num_timesteps - 1))

    def make_prediction_for_csv(self, frames):
        # TODO: resampling before input
        frames_preproc, _ = create_X(frames, self.preproc_params)
        frames_preproc = self.network.scaler.transform(frames_preproc)
        self.network.forward_prop(frames_preproc)
        self.prediction = softmax2one_hot(self.network.O[-1].T)

    def make_prediction_for_live(self, frames):
        if len(frames) == self.preproc_params.num_timesteps:
            # TODO: resampling before input
            frames_preproc, _ = create_X(frames, self.preproc_params)
            frames_preproc = self.network.scaler.transform(frames_preproc)
            self.network.forward_prop(frames_preproc)
            self.prediction = softmax2one_hot(self.network.O[-1].T)
        else:
            print("not enough frames yet")

    def compute_events(self, prediction: np.array):
        predicted_value = np.argmax(prediction)
        # print(predicted_value)
        self.iterated.append(predicted_value)
        self.iterated.pop(0)
        if predicted_value == 0:
            self.events.append("idle")
            if self.no_consider > 0 and self.iterated[-1] == 0:  #5x "idle" (NICHT direkt) hintereinander TODO
                self.no_consider -= 1
        elif not predicted_value == 0:
            if self.no_consider == 0:
                counter = self.iterated.count(predicted_value)
                if counter >= self.emitting_number:
                    self.events.append(self.dictionary[predicted_value])
                    # TODO: if Live Mode -> handle slideshow
                    print(self.dictionary[predicted_value])
                    # set counter to number of idles before a gesture can be detected:
                    self.no_consider = self.set_no_consider
                else:
                    #self.events.append("detected gesture")
                    self.events.append("idle")
            else:
                #self.events.append("gesture still happening")
                self.events.append("idle")

    def events_to_csv(self, frames:pd.DataFrame, output_path:str):
        events_df = pd.DataFrame(self.events)
        frames["events"] = events_df
        frames.set_index('timestamp', inplace=True)
        frames["events"].to_csv(output_path, index=True)
        print("events exported to %s" % output_path)


def create_Application():

    mediapipe_colums_for_diff = [
        # "left_shoulder_x", "left_shoulder_y",
        # "right_shoulder_x", "right_shoulder_y",
        "left_elbow_x", "left_elbow_y",
        "right_elbow_x", "right_elbow_y",
        "left_wrist_x", "left_wrist_y",
        "right_wrist_x", "right_wrist_y",
        # "left_index_x", "left_index_y", # "left_index_z",
        # "right_index_x", "right_index_y", # "right_index_z"
    ]
    mediapipe_columns_for_sum = mediapipe_colums_for_diff

    preproc_params = Preprocessing_parameters(
        num_shifts=1, num_timesteps=7,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
        summands_pattern=[1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    network_path = Path('../../saved_runs/first_run_max/2022-03-12_0_72-40-40-30-20-10-4')
    network = FCNN.load_run(network_path)

    my_model = Application(network, preproc_params)

    return my_model
