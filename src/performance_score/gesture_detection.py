import os
import argparse
import pandas as pd
import random
from pathlib import Path
from preprocessing.preprocessing_functions import Preprocessing_parameters, mediapipe_columns_for_sum, create_X
from preprocessing.live_preprocessing import LiveDfGenerator
from slideshow.prediction_functions_jonas_mod import create_PredictionHandler_for_test

# parser = argparse.ArgumentParser()
# parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
# parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

# args = parser.parse_known_args()[0]

# input_path = args.input_csv

# output_directory, input_csv_filename = os.path.split(args.input_csv)
# output_path = "%s/%s" % (output_directory, args.output_csv_name)

input_path = Path(r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\final_project_clone\final-project-getting-started\demo_data\demo_video_frames_rotate.csv')
input_path = Path(r'data\labeled_frames\tamara_val\rotate_labeled.csv')
output_path = input_path.parent / "emitted_events.csv"
output_path = Path(r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\project_dev_repo\ml_dev_repo\src\evaluation') / "emitted_events.csv"

frames = pd.read_csv(input_path, sep=' *,', engine='python')
frames.timestamp = frames.timestamp.astype(int)
# frames = frames.reindex()


# ================================= your application =============================
# you should import and call your own application here


def compute_events(frames: pd.DataFrame):
    pred_handler = create_PredictionHandler_for_test()
    org_frames = frames.copy()
    frames = frames.set_index('timestamp')
    df = LiveDfGenerator.resample(frames)
    # df = df[-pred_handler.preproc_params.num_timesteps:]
    # df = LiveDfGenerator.reset_index(df)

    
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