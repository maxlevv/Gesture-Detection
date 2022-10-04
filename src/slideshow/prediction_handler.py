import sys
sys.path.append('neural_net_pack')
sys.path.append('../../neural_net_pack')
import pandas as pd
import numpy as np
import time
from pathlib import Path
from neural_network import FCNN
from preprocessing.pca import PCA
from preprocessing.preprocessing_functions import Preprocessing_parameters
from preprocessing.preprocessing_functions import create_X
from preprocessing.preprocessing_functions import LabelsOptional
from helper import softmax2one_hot


class PredictionHandler():

    def __init__(self, network: FCNN, preproc_params: Preprocessing_parameters, observation_window: int = 30,
                 emitting_number: int = 5, set_no_consider: int = 10, labels=LabelsOptional, test_mode:bool =False, pca:PCA =None):
        self.network = network
        self.preproc_params = preproc_params
        self.labels = labels
        self.emitting_number = emitting_number
        self.set_no_consider = set_no_consider
        self.no_consider = 0
        self.iterated = [None] * observation_window
        self.prediction = None
        self.events = None
        self.rotate_emitting_number = 8
        if test_mode:
            self.swipe_emitting_number = 3
        else:
            self.swipe_emitting_number = emitting_number
        
        self.timestamp_for_events = None
        self.pca = pca

    def initialize_events(self, df=None):
        self.events = []
        self.events.extend(['idle'] * (self.preproc_params.num_timesteps - 1))
        if not df is None:
            self.timestamp_for_events = list(
                df.iloc[:self.preproc_params.num_timesteps - 1, df.columns.get_loc('timestamp')])

    def make_prediction_for_csv(self, resampled_df: pd.DataFrame):
        frames_preproc, _ = create_X(resampled_df, self.preproc_params)
        frames_preproc = self.network.scaler.transform(frames_preproc)
        # frames_preproc = self.pca.transform(X=frames_preproc)
        self.network.forward_prop(frames_preproc)
        self.prediction = softmax2one_hot(self.network.O[-1].T)

    def make_prediction_for_live(self, resampled_df: pd.DataFrame) -> np.array:
        frames_preproc, _ = create_X(resampled_df, self.preproc_params)
        frames_preproc = self.network.scaler.transform(frames_preproc)
        self.network.forward_prop(frames_preproc)
        self.prediction = softmax2one_hot(self.network.O[-1].T)
        return self.prediction

    def compute_events(self, prediction: np.array, timestamp=None) -> str:
        predicted_value = np.argmax(prediction)
        self.iterated.append(predicted_value)
        self.iterated.pop(0)
        if predicted_value == 0:
            self.events.append("idle")
            # 5x "idle" (NICHT direkt) hintereinander could be added here
            if self.no_consider > 0 and self.iterated[-2] == 0:
                self.no_consider -= 1
        elif not predicted_value == 0:
            if self.no_consider == 0:
                if predicted_value == LabelsOptional.rotate.value or predicted_value == LabelsOptional.rotate_left.value:
                    lokal_emitting_number = self.rotate_emitting_number
                elif predicted_value == LabelsOptional.swipe_left.value or predicted_value == LabelsOptional.swipe_right.value:
                    lokal_emitting_number = self.swipe_emitting_number
                else:
                    lokal_emitting_number = self.emitting_number
                counter = self.iterated.count(predicted_value)
                if counter >= lokal_emitting_number:
                    self.events.append(self.labels(predicted_value).name)

                    # set counter to number of idles before a gesture can be detected:
                    self.no_consider = self.set_no_consider
                else:
                    self.events.append("idle")
            else:
                self.events.append("idle")

        if not timestamp is None:
            self.timestamp_for_events.append(timestamp)
        return self.events[-1]

    def events_to_csv(self, frames: pd.DataFrame, output_path: str):
        frames.index = frames.index.astype(int)

        pred_int = np.argmax(self.prediction, axis=1)
        pred_int = np.concatenate([np.zeros(9), pred_int], axis=0)

        # events die zugewiesen werden müssen
        events_df = pd.DataFrame(
            {'timestamp': self.timestamp_for_events, 'events': self.events, 'predictions': pred_int})

        # alle timestamp nuller
        frames_zero_timestamp = frames[frames['timestamp'] == 0].copy()

        # falls der erste timestamp nuller auch index 0 hat, dann soll dieser ausgelassen werden
        if frames_zero_timestamp.iloc[0]['timestamp'] == 0:
            frames_zero_timestamp = frames_zero_timestamp[1:]

        # den nullern idle zuweisen
        frames_zero_timestamp.loc[:, 'events'] = 'idle'

        # nicht nuller bestimmen
        frames_none_zero_timestamp = frames.loc[:
                                                frames_zero_timestamp.index[0] - 1]

        frames_none_zero_timestamp.loc[:, 'events'] = 'idle'

        # looking vor non idle events and putting them in the frames df at the next largest timestamp
        for i in range(len(events_df)):
            event = events_df.loc[i, 'events']
            timestamp_of_event = events_df.loc[i, 'timestamp']
            if event != 'idle':
                timestamp_larger_then_timestamp_of_event_condition = frames_none_zero_timestamp[
                    'timestamp'] >= timestamp_of_event
                try:
                    index_to_write_event_in = timestamp_larger_then_timestamp_of_event_condition[
                        timestamp_larger_then_timestamp_of_event_condition].index[0]
                    frames_none_zero_timestamp.loc[index_to_write_event_in,
                                                   'events'] = event
                except IndexError as e:
                    print(
                        'there was no timestamp larger then the one given in events_df for a non idle event!!!!!!!!!!')

        # combine both back together
        frames_res = pd.concat([frames_none_zero_timestamp[[
                               'timestamp', 'events']], frames_zero_timestamp[['timestamp', 'events']]], axis=0)

        frames.set_index('timestamp', inplace=True)
        frames["events"] = list(frames_res['events'])

        frames["events"].to_csv(output_path, index=True)


def create_PredictionHandler_for_live():

    mediapipe_colums_for_diff = [
        "left_elbow_x", "left_elbow_y",
        "right_elbow_x", "right_elbow_y",
        "left_wrist_x", "left_wrist_y",
        "right_wrist_x", "right_wrist_y",
    ]
    mediapipe_columns_for_sum = mediapipe_colums_for_diff

    preproc_params = Preprocessing_parameters(
        # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
        num_shifts=1, num_timesteps=10,
        summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    network_path = Path(
        '../../saved_runs/jonas_final_gross/relu,ep=700,bs=512,lr=0.000875,wd=0/2022-03-31_2_110-40-40-30-20-11')

    network = FCNN.load_run(network_path)

    pred_handler = PredictionHandler(network, preproc_params)

    return pred_handler


def create_PredictionHandler_for_test():

    mediapipe_colums_for_diff = [
        "left_elbow_x", "left_elbow_y",
        "right_elbow_x", "right_elbow_y",
        "left_wrist_x", "left_wrist_y",
        "right_wrist_x", "right_wrist_y",
    ]
    mediapipe_columns_for_sum = mediapipe_colums_for_diff

    preproc_params = Preprocessing_parameters(
        num_shifts=1, num_timesteps=10,
        summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    network_path = Path(r'..\..\saved_runs\kleines_netz,new_window=10,pattern=all_600epochs\relu,ep=600,bs=512,lr=0.000875,wd=0\2022-03-31_2_110-30-30-4')
    network_path = Path(r'..\..\saved_runs\kleines_netz,new_window=10,pattern=all_1000epochs\relu,ep=1000,bs=512,lr=0.000875,wd=0\2022-04-01_2_110-30-30-15-4')
    network_path = Path(r'..\..\saved_runs\kleines_netz,new_window=10,pattern=all_600epochs\relu,ep=600,bs=512,lr=0.000875,wd=0\2022-03-31_0_110-30-30-15-4')
    

    network = FCNN.load_run(network_path)

    # pca = PCA.load(r'..\..\data\preprocessed_frames\new_window=10,cumsum=all\pca_mandatory.json')

    pred_handler = PredictionHandler(network, preproc_params, test_mode=True)

    return pred_handler
