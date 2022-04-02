import pandas as pd
import numpy as np
import time
from pathlib import Path
from modeling.neural_network import FCNN
from preprocessing.preprocessing_functions import Preprocessing_parameters
from preprocessing.preprocessing_functions import create_X
from preprocessing.preprocessing_functions import LabelsOptional
from modeling.helper import softmax2one_hot


class PredictionHandler():

    def __init__(self, network: FCNN, preproc_params: Preprocessing_parameters, observation_window: int = 30,
                 emitting_number: int = 5, set_no_consider: int = 5, labels=LabelsOptional):
        self.network = network
        self.preproc_params = preproc_params
        # self.dictionary = {0: 'idle', 1: 'swipe_right', 2: 'swipe_left', 3: 'rotate'}
        self.labels = labels
        self.emitting_number = emitting_number
        self.set_no_consider = set_no_consider
        self.no_consider = 0
        self.iterated = [None] * observation_window
        #self.iterated = deque(maxlen=observation_window)
        self.prediction = None
        self.events = None
        self.timestamp_for_events = None

    def initialize_events(self, df=None):
        self.events = []
        #self.events.extend(['buffer'] * (self.preproc_params.num_timesteps - 1))
        self.events.extend(['idle'] * (self.preproc_params.num_timesteps - 1))
        if not df is None:
            self.timestamp_for_events = list(df.iloc[:self.preproc_params.num_timesteps - 1, df.columns.get_loc('timestamp')])

    def make_prediction_for_csv(self, resampled_df: pd.DataFrame):
        frames_preproc, _ = create_X(resampled_df, self.preproc_params)
        frames_preproc = self.network.scaler.transform(frames_preproc)
        self.network.forward_prop(frames_preproc)
        self.prediction = softmax2one_hot(self.network.O[-1].T)

    def make_prediction_for_live(self, resampled_df: pd.DataFrame) -> np.array: 
        # print(resampled_df)
        # print("net_size", self.preproc_params.num_timesteps)
        # t1 = time.perf_counter()
        frames_preproc, _ = create_X(resampled_df, self.preproc_params)
        # print('create_x time', time.perf_counter() - t1)
        frames_preproc = self.network.scaler.transform(frames_preproc)
        self.network.forward_prop(frames_preproc)
        self.prediction = softmax2one_hot(self.network.O[-1].T)
        # print('self.prediction', self.prediction)
        # print(self.prediction[0][0])
        # if not self.prediction[0][0] == 1:
        #     print('confidence' , self.network.O[-1].T[self.prediction.astype(bool)])
        # print('confidence' , self.network.O[-1].T[self.prediction])
        return self.prediction

    def compute_events(self, prediction: np.array, timestamp=None) -> str:
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
                    # self.events.append(self.dictionary[predicted_value])
                    self.events.append(self.labels(predicted_value).name)
                    # print('confidence' , self.network.O[-1].T[self.prediction.astype(bool)])
                    # print(self.labels(predicted_value).name)
                    # set counter to number of idles before a gesture can be detected:
                    self.no_consider = self.set_no_consider
                else:
                    #self.events.append("detected gesture")
                    self.events.append("idle")
            else:
                #self.events.append("gesture still happening")
                self.events.append("idle")

        if not timestamp is None:
            self.timestamp_for_events.append(timestamp)
        return self.events[-1]

    def events_to_csv(self, frames:pd.DataFrame, output_path:str):
        orig_frames = frames.copy()
        frames.index = frames.index.astype(int)

        # events die zugewiesen werden müssen
        events_df = pd.DataFrame({'timestamp' : self.timestamp_for_events, 'events' : self.events})

        # alle timestamp nuller
        frames_zero_timestamp = frames[frames['timestamp'] == 0].copy()

        # falls der erste timestamp nuller auch index 0 hat, dann soll dieser ausgelassen werden
        if frames_zero_timestamp.iloc[0]['timestamp'] == 0:
            frames_zero_timestamp = frames_zero_timestamp[1:]

        # den nullern idle zuweisen
        frames_zero_timestamp.loc[:, 'events'] = 'idle'

        # nicht nuller bestimmen
        frames_none_zero_timestamp = frames.loc[:frames_zero_timestamp.index[0] - 1]

        frames_none_zero_timestamp.loc[:, 'events'] = 'idle'

        # looking vor non idle events and putting them in the frames df at the next largest timestamp
        for i in range(len(events_df)):
            event = events_df.loc[i, 'events']
            timestamp_of_event = events_df.loc[i, 'timestamp']
            if event != 'idle':
                timestamp_larger_then_timestamp_of_event_condition = frames_none_zero_timestamp['timestamp'] >= timestamp_of_event
                try:
                    index_to_write_event_in = timestamp_larger_then_timestamp_of_event_condition[timestamp_larger_then_timestamp_of_event_condition].index[0]
                    frames_none_zero_timestamp.loc[index_to_write_event_in, 'events'] = event
                except IndexError as e:
                    print('there was no timestamp larger then the one given in events_df for a non idle event!!!!!!!!!!')
                

        # combine both back together
        frames_res = pd.concat([frames_none_zero_timestamp[['timestamp', 'events']], frames_zero_timestamp[['timestamp', 'events']]], axis=0)

        frames.set_index('timestamp', inplace=True)
        frames["events"] = list(frames_res['events'])
        
        frames["events"].to_csv(output_path, index=True)
        


def create_PredictionHandler_for_live():

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
        num_shifts=1, num_timesteps=10,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
        summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    network_path = Path(r'../../saved_runs\first_run_max\2022-03-12_0_72-40-40-30-20-10-4')
    network_path = Path(r'../../saved_runs\try_live_models\leaky_relu,ep=80,bs=512,lr=0.004814,wd=0.004551\2022-03-28_0_112-40-40-30-20-10-11')
    network_path = Path(r'../..\saved_runs\jonas_random_4\relu,ep=800,bs=64,lr=0.000857,wd=0\2022-03-30_0_88-40-40-20-10-11')
    network_path = Path(r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\project_dev_repo\ml_dev_repo\saved_runs\test_live_mode_run\relu,ep=700,bs=512,lr=0.000875,wd=0\2022-03-31_0_110-30-30-30-11')

    network = FCNN.load_run(network_path)

    pred_handler = PredictionHandler(network, preproc_params)

    return pred_handler


def create_PredictionHandler_for_test():

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
        num_shifts=1, num_timesteps=10,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
        summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    
    network_path = Path(r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\project_dev_repo\ml_dev_repo\saved_runs\test_live_mode_run\ml_dev_repo-max-dev-saved_runs-kleines_netz,new_window=10,pattern=all_600epochs\saved_runs\kleines_netz,new_window=10,pattern=all_600epochs\relu,ep=600,bs=512,lr=0.000875,wd=0\2022-03-31_2_110-30-30-4')

    network = FCNN.load_run(network_path)

    pred_handler = PredictionHandler(network, preproc_params)

    return pred_handler
