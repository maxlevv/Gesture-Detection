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


FLIP_BOOL = True

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


correct_columns = ['timestamp', 'nose_x', 'nose_y', 'nose_z', 'nose_confidence', 'left_eye_inner_x', 'left_eye_inner_y', 
                   'left_eye_inner_z', 'left_eye_inner_confidence', 'left_eye_x', 'left_eye_y', 'left_eye_z', 'left_eye_confidence', 
                   'left_eye_outer_x', 'left_eye_outer_y', 'left_eye_outer_z', 'left_eye_outer_confidence', 'right_eye_inner_x', 
                   'right_eye_inner_y', 'right_eye_inner_z', 'right_eye_inner_confidence', 'right_eye_x', 'right_eye_y', 'right_eye_z', 
                   'right_eye_confidence', 'right_eye_outer_x', 'right_eye_outer_y',
                   'right_eye_outer_z', 'right_eye_outer_confidence', 'left_ear_x', 'left_ear_y', 'left_ear_z', 
                   'left_ear_confidence', 'right_ear_x', 'right_ear_y', 'right_ear_z', 'right_ear_confidence', 'left_mouth_x', 
                   'left_mouth_y', 'left_mouth_z', 'left_mouth_confidence', 'right_mouth_x', 'right_mouth_y', 'right_mouth_z', 
                   'right_mouth_confidence', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_shoulder_confidence', 
                   'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_shoulder_confidence', 'left_elbow_x', 
                   'left_elbow_y', 'left_elbow_z', 'left_elbow_confidence', 'right_elbow_x', 'right_elbow_y', 'right_elbow_z', 
                   'right_elbow_confidence', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z', 'left_wrist_confidence', 'right_wrist_x', 
                   'right_wrist_y', 'right_wrist_z', 'right_wrist_confidence', 'left_pinky_x', 'left_pinky_y', 'left_pinky_z', 
                   'left_pinky_confidence', 'right_pinky_x', 'right_pinky_y', 'right_pinky_z', 'right_pinky_confidence', 'left_index_x', 
                   'left_index_y', 'left_index_z', 'left_index_confidence', 'right_index_x', 'right_index_y', 'right_index_z', 
                   'right_index_confidence', 'left_thumb_x', 'left_thumb_y', 'left_thumb_z', 'left_thumb_confidence', 'right_thumb_x', 
                   'right_thumb_y', 'right_thumb_z', 'right_thumb_confidence', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_hip_confidence', 
                   'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_confidence', 'left_knee_x', 'left_knee_y', 'left_knee_z', 
                   'left_knee_confidence', 'right_knee_x', 'right_knee_y', 'right_knee_z', 'right_knee_confidence', 'left_ankle_x', 
                   'left_ankle_y', 'left_ankle_z', 'left_ankle_confidence', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 
                   'right_ankle_confidence', 'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_heel_confidence', 'right_heel_x', 
                   'right_heel_y', 'right_heel_z', 'right_heel_confidence', 'left_foot_index_x', 'left_foot_index_y', 
                   'left_foot_index_z', 'left_foot_index_confidence', 'right_foot_index_x', 'right_foot_index_y', 
                   'right_foot_index_z', 'right_foot_index_confidence']


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


def flip(frames: pd.DataFrame):
    flipped_df = pd.DataFrame()
    for correct_column in correct_columns:
        corresponding_frames_column = correct_column
        if 'right' in correct_column:
            corresponding_frames_column = correct_column.replace('right', 'left')
        if 'left' in correct_column:
            corresponding_frames_column = correct_column.replace('left', 'right')
        if '_x' in correct_column:
            # transform x values with 1-x
            flipped_df[correct_column] = 1 - frames[corresponding_frames_column].rename(correct_column)
        else:
            # dont transform y values or others like timestamp
            flipped_df[correct_column] = frames[corresponding_frames_column].rename(correct_column)
        
    
    return flipped_df


if FLIP_BOOL:
    flipped_frames = flip(frames)
else:
    flipped_frames = frames


compute_events(flipped_frames)

# ================================================================================


print("events exported to %s" % output_path)


# if __name__ == '__main__':
#     run_gesture_detection(mode='test')
