import sys
sys.path.append('..')
from pathlib import Path
import random
import pandas as pd
import numpy as np
import argparse
import os

from slideshow.prediction_handler import create_PredictionHandler_for_test
from preprocessing.live_preprocessing import LiveDfGenerator
from preprocessing.preprocessing_functions import Preprocessing_parameters, mediapipe_columns_for_sum, create_X


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name",
                    help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)


# input_path = Path(r'..\..\data\raw_frames\tamara_val_original_to_csv\swipe_right_2022-04-03_14-55.csv')
# output_path = input_path.parent / "emitted_events_swipe_right.csv"
# output_path = Path(
#     r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\project_dev_repo\ml_dev_repo\src\evaluation') / "emitted_events.csv"

frames = pd.read_csv(input_path, sep=' *,', engine='python')
frames.timestamp = frames.timestamp.astype(int)


def compute_events(frames: pd.DataFrame):
    pred_handler = create_PredictionHandler_for_test()
    org_frames = frames.copy()
    frames = frames.set_index('timestamp')
    df = LiveDfGenerator.resample(frames)

    pred_handler.make_prediction_for_csv(df)
    pred_handler.initialize_events(df)
    timestamp_list = list(df['timestamp'])[len(pred_handler.events):]
    for prediction, timestamp in zip(pred_handler.prediction, timestamp_list):
        pred_handler.compute_events(prediction, timestamp)

    pred_handler.events_to_csv(org_frames, output_path)



compute_events(frames)

# ================================================================================


print("events exported to %s" % output_path)


# if __name__ == '__main__':
#     run_gesture_detection(mode='test')
