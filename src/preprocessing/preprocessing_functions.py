import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from enum import Enum
import glob

# preprocessing parameters

# considering only the manitroy gestures (swipe right, left, rotation)

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


class Labels(Enum):
    idle = 0
    swipe_right = 1
    swipe_left = 2
    rotate = 3

    def get_label_list() -> List[str]:
        # currently not used
        return ['idle', 'swipe_right', 'swipe_left', 'rotate']

    def get_column_names() -> List[str]:
        # this is the notation of the one hot encoded ground truth columns in the dataframe
        return ['gt_' + label_abbreviation for label_abbreviation in ['idle', 'sr', 'sl', 'r']]


class Preprocessing_parameters():
    def __init__(self, num_shifts: int, num_timesteps: int, 
                difference_mode: str = None, mediapipe_columns_for_diff: List[str] = None,
                summands_pattern: List[int] = None, media_pipe_columns_for_sum: List[str] = None):
        self.num_shifts = num_shifts
        self.num_timesteps = num_timesteps
        self.difference_mode = difference_mode
        self.mediapipe_colums_for_diff = mediapipe_columns_for_diff
        self.summands_pattern = summands_pattern
        self.mediapipe_columns_for_sum = media_pipe_columns_for_sum


def extract_features(frames: pd.DataFrame, features: list) -> pd.DataFrame:
    # features = features.copy()
    # features.append("ground_truth")
    return frames.loc[:, features]


def preprocessing_difference(frames, number_timestamps: int, number_shifts: int):
    m = int(frames.shape[0] / number_shifts - number_timestamps)
    n = frames.shape[1] - 1

    # data matrix to be returned
    X = np.array(np.zeros((m, n)))

    # ground truth vector to be returned, for now m x 1 (only rotate or not)
    y = np.array(np.zeros(m))

    for i in range(m):
        data = ((frames.iloc[:, :-1]).iloc[i * number_shifts: (i *
                number_shifts) + number_timestamps]).to_numpy()
        difference = data[number_timestamps - 1, :] - data[0]
        X[i, :] = difference

        if frames["ground_truth"].iloc[(i * number_shifts) + number_timestamps] != "idle":
            y[i] = 1

    # fig, ax = plt.subplots(figsize=(12, 10))
    # ax.plot(X[0:, 1])
    # ax.plot(X[0:, 0])
    # ax.plot(y / 6)
    # plt.show()
    return X, y


def calc_differences(df: pd.DataFrame, preproc_params: Preprocessing_parameters) -> Tuple[np.array, pd.DataFrame]:
    """ Calculates the features given as differences from all the columns in df considering a shift of num_shifts
        after each difference and considering num_timesteps timesteps for each row in the output.
        Difference mode determines the way the differences are calculated, e.g. 'every' is adding all the differences between timesteps 
        in the num_timesteps window and 'one' is using only the diff between the last and first timestep in the num_timesteps window

    Args:
        df (pd.DataFrame): needs to have only the columns which should be operated on 
        num_timesteps (int): timesteps for the calculation of each row in the output
        num_shifts (int): shift after each dif
        difference_mode (str): _description_

    Returns:
        Tuple[np.array, pd.DataFrame]: result 
    """

    num_timesteps = preproc_params.num_timesteps
    num_shifts = preproc_params.num_shifts

    # necessary as otherwise df is a view and the drop affects the original df
    df = df.copy()

    # number of samples in X
    num_samples = math.floor(
        (df.shape[0] - num_timesteps + num_shifts) / num_shifts)

    data = df.to_numpy()
    orig_columns = df.columns

    if preproc_params.difference_mode == 'one':
        # diff between last and first of current sliding window

        num_features = data.shape[1]

        X = np.zeros((num_samples, num_features))

        for i in range(num_samples):
            X[i, :] = data[i * num_shifts + num_timesteps - 1, :] - \
                data[i * num_shifts, :]

        X_df = pd.DataFrame(
            data=X, columns=[orig_column + "_diff" for orig_column in orig_columns])

    elif preproc_params.difference_mode == 'every':
        # diff between every element of the current slinding window

        # there are (num_timesteps - 1) differences
        num_differences_in_a_sample = num_timesteps - 1
        num_features = data.shape[1] * num_differences_in_a_sample

        X = np.zeros((num_samples, num_features))

        # calcing all the consecutive diffs
        diff_of_consecutive_rows = data[1:] - data[:-1, :]

        # placing the diffs in X
        for i in range(num_samples):
            X[i, :] = diff_of_consecutive_rows[i*num_shifts: i *
                                               num_shifts + num_differences_in_a_sample, :].flatten('F').reshape(1, -1)

        # creating the df
        column_diff_names = [orig_column +
                             "_diff" for orig_column in orig_columns]
        X_df = pd.DataFrame(data=X, columns=[
                            column_diff_name + f"_{str(i)}" for column_diff_name in column_diff_names for i in range(num_differences_in_a_sample)])

    return X, X_df


def cumulative_sum(df: pd.DataFrame, preproc_params: Preprocessing_parameters) -> Tuple[np.array, pd.DataFrame]:

    num_timesteps = preproc_params.num_timesteps
    num_shifts = preproc_params.num_shifts
    summands_pattern = preproc_params.summands_pattern

    # necessary as otherwise df is a view and the drop affects the original df
    df = df.copy()

    # number of samples in X
    num_samples = math.floor(
        (df.shape[0] - num_timesteps + num_shifts) / num_shifts)

    data = df.to_numpy()
    orig_columns = df.columns

    num_features = data.shape[1] * sum(summands_pattern)

    X = np.zeros((num_samples, num_features))

    for i in range(num_samples):
        window_start_index = i * num_shifts
        window_np = data[window_start_index: window_start_index + num_timesteps, :]

        # calc the cumulative sum over the whole window
        cumsum = np.cumsum(window_np, axis=0)

        # extract only the cumsums specified in the pattern
        cumsum_features = cumsum[np.array(summands_pattern, dtype=bool), :]

        # adding the features to X
        X[i, :] = cumsum_features.flatten('F').reshape(1, -1)

    # creating the df
    column_cumsum_names = [orig_column +
                            "_cumsum" for orig_column in orig_columns]
    X_df = pd.DataFrame(data=X, columns=[
                        column_cumsum_name + f"_{str(i)}" for column_cumsum_name in column_cumsum_names for i in range(sum(summands_pattern))])

    return X, X_df


def determine_label_from_ground_truth_vector(ground_truth_df: pd.DataFrame, num_timesteps: int, num_shifts: int) -> np.array:
    num_samples = math.floor(
        (ground_truth_df.shape[0] - num_timesteps + num_shifts) / num_shifts)

    y = np.zeros((num_samples, ground_truth_df.shape[1]))

    data = ground_truth_df.to_numpy()
    for i in range(num_samples):
        y[i, :] = data[i*num_shifts, :]

    return y


def add_one_hot_encoding_to_df(y: np.array, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Args:
        y (np.array): one_hot_encoded vector
        df (pd.DataFrame, optional): df to add the one_hot_encoded columns to. 
            Defaults to None, which results in returning a new df
    """

    y_df = pd.DataFrame(
        data=y, columns=Labels.get_column_names(), dtype=np.int16)
    if df is not None:
        df = pd.concat([df, y_df], axis=1)
    else:
        df = y_df
    return df


def replace_str_label_by_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): needs to have the str label in the 'ground_truth' column
    """
    # turn str label into int
    def ground_truth_str_to_int_mapping(x): return Labels[x].value
    df['ground_truth_int'] = df['ground_truth']
    df['ground_truth_int'] = df['ground_truth_int'].apply(
        ground_truth_str_to_int_mapping)

    # create one hot encoding
    # note that this is not the y desired in the end, as the rows are not the samples
    y = np.identity(len(Labels), dtype=int)[
        df['ground_truth_int'].to_numpy(dtype=int)]

    df = add_one_hot_encoding_to_df(y, df)

    df.drop(['ground_truth', 'ground_truth_int'], inplace=True, axis=1)

    return df


def create_X(df: pd.DataFrame, preproc_params: Preprocessing_parameters) -> Tuple[np.array, pd.DataFrame]:
    # num_shifts: int, num_timesteps: int, difference_mode: str, summands_pattern: List[int]) -> Tuple[np.array, pd.DataFrame]:
    # evtl add fuctions for other features here

    # number of samples in X
    num_samples = math.floor(
        (df.shape[0] - preproc_params.num_timesteps + preproc_params.num_shifts) / preproc_params.num_shifts)

    if preproc_params.difference_mode:
        df_for_diff = extract_features(df, preproc_params.mediapipe_colums_for_diff)

        X_diff, X_diff_df = calc_differences(df_for_diff, preproc_params)
    else:
        X_diff, X_diff_df = np.array([]).reshape(num_samples, 0), None

    if preproc_params.summands_pattern:
        df_for_sum = extract_features(df, preproc_params.mediapipe_columns_for_sum)

        X_sum, X_sum_df = cumulative_sum(df_for_sum, preproc_params)
    else:
        X_sum, X_sum_df = np.array([]).reshape(num_samples, 0), None

    return np.c_[X_diff, X_sum].round(4), pd.concat([X_diff_df, X_sum_df], axis=1).round(4)


def create_y(df: pd.DataFrame, preproc_params : Preprocessing_parameters) -> Tuple[np.array, pd.DataFrame]:
    y = determine_label_from_ground_truth_vector(
        df[Labels.get_column_names()], preproc_params.num_timesteps, preproc_params.num_shifts)
    y_df = add_one_hot_encoding_to_df(y)
    return y, y_df


def preprocessing(labeled_frame_file_path: Path, preproc_params: Preprocessing_parameters) \
        -> Tuple[np.array, pd.DataFrame]:
        # num_shifts: int, num_timesteps: int,
        # difference_mode: str = None, mediapipe_colums_for_diff: List[str] = None, 
        # summands_pattern: List[int] = None, mediapipe_colums_for_sum: List[str] = None) \
        
    """Takes path of a labeled df and preprocesses it/ creates the features for the input of the neural network

    Args:
        mediapipe_colums_for_diff (List[str]): all column names to calc the differences of
        num_shifts (int): how far the sliding window for calcing the diffs is shifted after calcing one diff
        num_timesteps (int): how many timesteps are in a sliding window for calcing the diff
        difference_mode (str, optional): how the differences are calculated. Defaults to 'one'.

    Returns:
        _type_: _description_
    """
    labeled_df = pd.read_csv(labeled_frame_file_path, sep=' *,', engine='python')
    # df = extract_features(labeled_df, mediapipe_colums_for_diff)
    # del labeled_df
    # now the one hot encoding is added to the df and later take out in create y, which is kind of unnecessary, but this functionality might be usefull for something else
    df = replace_str_label_by_one_hot_encoding(labeled_df)
    X, X_df = create_X(df, preproc_params)
    y, y_df = create_y(df, preproc_params)
    nn_input = (X, y)
    nn_input_df = pd.concat([X_df, y_df], axis=1)

    return nn_input, nn_input_df


def handle_preprocessing(labeled_frames_folder_path: Path, preprocessed_frames_folder_path: Path, preproc_params: Preprocessing_parameters):
    """gets all '*_labeled.csv' files under the specified folder, does preprocessing with the specified parameters and saves it in the other specified folder.
    Args:
        labeled_frames_folder_path (Path): load folder topath
        preprocessed_frames_folder_path (Path): to folder path
    """
    for labeled_csv_file_path in labeled_frames_folder_path.glob('**/*_labeled.csv'):
        _, nn_input_df = preprocessing(labeled_csv_file_path, preproc_params)

        nn_input_df.to_csv(preprocessed_frames_folder_path /
                           labeled_csv_file_path.name.replace("_labeled.csv", "_preproc.csv"))



if __name__ == '__main__':

    # nn_input_df.to_csv('nn_input_test.csv')

    preproc_params = Preprocessing_parameters(
        num_shifts=2, num_timesteps=5, difference_mode="one",
        mediapipe_columns_for_diff=mediapipe_colums_for_diff)
    LABEL_PATH = Path("..\..\data\labeled_frames")
    PREPROC_PATH = Path(r"..\..\data\preprocessed_frames")

    handle_preprocessing(LABEL_PATH, PREPROC_PATH, preproc_params)

    print('done')
