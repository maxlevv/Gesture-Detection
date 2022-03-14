import os
import argparse
import pandas as pd
from pathlib import Path
from modeling.neural_network import FCNN
from preprocessing.preprocessing_functions import Preprocessing_parameters
from slideshow.prediction_functions import Application

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

frames = pd.read_csv(input_path, sep=' *,', engine='python')  # set_index 'timestamp' done in Application

# ================================= your application =============================
# you should import and call your own application here

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

network_path = Path(r'../../saved_runs\first_run_max\2022-03-12_0_72-40-40-30-20-10-4')
network = FCNN.load_run(network_path)

my_model = Application(network, preproc_params)
my_model.make_prediction_for_csv(frames)
my_model.initialize_events()
for prediction in my_model.prediction:
    my_model.compute_events(prediction)

my_model.events_to_csv(frames, output_path)

# ================================================================================
