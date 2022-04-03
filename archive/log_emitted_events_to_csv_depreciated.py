import os
import argparse
import pandas as pd
from slideshow.prediction_functions import create_Application

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", help="CSV file containing the video transcription from OpenPose", required=True)
parser.add_argument("--output_csv_name", help="output CSV file containing the events", default="emitted_events.csv")

args = parser.parse_known_args()[0]

input_path = args.input_csv

output_directory, input_csv_filename = os.path.split(args.input_csv)
output_path = "%s/%s" % (output_directory, args.output_csv_name)

frames = pd.read_csv(input_path, sep=' *,', engine='python')  # set_index 'timestamp' done in Application
print(frames)
# ================================= your application =============================
# you should import and call your own application here

my_model = create_Application()
my_model.make_prediction_for_csv(frames)
my_model.initialize_events()
for prediction in my_model.prediction:
    my_model.compute_events(prediction)

my_model.events_to_csv(frames, output_path)

# ================================================================================
