import time
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from process_videos.helpers.colors import bcolors


class LiveDfGenerator:
    def __init__(self, relevant_signals_dict_yaml_path, window_size, skip_value=1):
        self.frame_list = []
        self.timestamps = []
        with open(relevant_signals_dict_yaml_path, "r") as yaml_file:
            self.mediapipe_tracking_points_dict = yaml.safe_load(yaml_file)
        self.column_names = self.determine_column_names()
        self.window_size = window_size

        self.skip_value = skip_value
        self.skip_status = 0

    
    def read_data(self, data, timestamp):
        if len(self.timestamps) > 0:
            if timestamp == self.timestamps[-1]:
                return -1
        frame = []
        if data is None:
            return None

        # only get the ones in the yaml file not all 33 like in original file     
        for i in list(self.mediapipe_tracking_points_dict.values()):
            frame.append(data.landmark[i].x)
            frame.append(data.landmark[i].y)
            frame.append(data.landmark[i].z)
            frame.append(data.landmark[i].visibility)
        self.frame_list.append(frame)
        self.timestamps.append(timestamp)
        return None

    def remove_oldest_frame_and_timestamp(self):
        self.frame_list.pop(0)
        self.timestamps.pop(0)

    def to_df(self):
        frames = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
        # print('frames in to_df', frames)
        frames.index.name = "timestamp"
        frames.index = frames.index.astype(int)
        return frames.round(5)
    
    @classmethod
    def resample(cls, df: pd.DataFrame):
        # the df should have timestamp as the index, in int format

        # src: https://stackoverflow.com/questions/47148446/pandas-resample-interpolate-is-producing-nans
        #   and https://pandas.pydata.org/docs/reference/api/pandas.timedelta_range.html#pandas.timedelta_range

        df.index = pd.to_timedelta(df.index, unit="ms")
        old_index = df.index
        new_index = pd.timedelta_range(old_index.min(), old_index.max(), freq='33ms')

        # this was added for test mode, but should not conflict with live mode
        df = df[~df.index.duplicated(keep='first')]

        df = df.reindex(old_index.union(new_index).drop_duplicates())
        df = df.interpolate()
        df = df.reindex(new_index)
        df = cls.make_index_look_nice(df)
        # print('df type before reset index', type(df))
        df = cls.reset_index(df)

        # print('df in resampling', df)
        # print('df type in resampling', type(df))
        # print('df columns in resampling', df.columns)
        # df.to_csv('resampled_df_to_reset_index.csv')
        
        return df

    @classmethod
    def make_index_look_nice(cls, df):
        df.index = df.index - list(df.index)[0]
        df.index = df.index.astype(int)
        df.index = df.index * 1e-6
        df.index = df.index.astype(int) 
        return df
    
    @classmethod
    def reset_index(cls, df):
        df = df.reset_index()
        df.columns = [column.replace('index', 'timestamp') for column in df.columns]
        return df
    
    def generate_window_df(self, new_data, timestamp, frame=None) -> pd.DataFrame:
        # print('frame in generate window df', frame)
        if frame:
            # print('frame appended')
            self.frame_list.append(frame)
            self.timestamps.append(timestamp)
        else:
            if self.read_data(new_data, timestamp) == -1:
                return -1

        if len(self.frame_list) < self.window_size:
            # in this case not enough farmes have been read yet, so it returns None
            # print('skip no enough frames')
            return None

        skip_time = time.perf_counter()
        if self.skip_status >= self.skip_value:
            self.skip_status = 0

        if self.skip_status > 0:
            self.remove_oldest_frame_and_timestamp()
            self.skip_status += 1
            # print(f'skip time: {time.perf_counter() - skip_time}')
            return None
        self.skip_status += 1
        
        df = self.to_df()

        # print('df in generate window df before resampling', df)

        # diff = np.diff(np.array(list(df.index)))
        # print(f"{bcolors.WARNING}{diff}{bcolors.ENDC}")
        
        df = self.resample(df)
        output_df = df[-self.window_size:]
        output_df = self.reset_index(output_df)
        self.remove_oldest_frame_and_timestamp()
        return output_df
        

    def determine_column_names(self):
        return ["%s_%s" % (joint_name, jdn) for joint_name in list(self.mediapipe_tracking_points_dict.keys()) for jdn in
                ["x", "y", "z", "confidence"]]


def test_resample():
    df = pd.read_csv(Path(r'..\..\data\raw_frames\test\2022-03-08 08-25-04_2022-03-08_08-36-53_raw.csv'))
    df = df.loc[:5, :]
    # df.timestamp = pd.to_datetime(df.timestamp, unit='ms')
    
    
    print('done')

if __name__ == '__main__':
    test_resample()