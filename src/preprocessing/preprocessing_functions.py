import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from enum import Enum
import glob
from tqdm import tqdm
from process_videos.helpers.colors import bcolors

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
        return [str(label.name) for label in Labels]

    def get_column_names() -> List[str]:
        # this is the notation of the one hot encoded ground truth columns in the dataframe
        return ['gt_' + label_abbreviation for label_abbreviation in ['idle', 'sr', 'sl', 'r']]

class LabelsMandatory(Enum):
    idle = 0
    swipe_right = 1
    swipe_left = 2
    rotate = 3

    def get_label_list() -> List[str]:
        # currently not used
        return [str(label.name) for label in LabelsMandatory]

    def get_column_names() -> List[str]:
        # this is the notation of the one hot encoded ground truth columns in the dataframe
        return ['gt_' + label_abbreviation for label_abbreviation in ['idle', 'sr', 'sl', 'r']]


class LabelsOptional(Enum):
    idle = 0
    swipe_right = 1
    swipe_left = 2
    rotate = 3
    rotate_left = 4
    swipe_up = 5
    swipe_down = 6
    pinch = 7
    spread = 8
    flip_table = 9
    point = 10

    def get_label_list() -> List[str]:
        # currently not used
        return [str(label.name) for label in LabelsOptional]

    def get_column_names() -> List[str]:
        return ['gt_' + label_abbreviation for label_abbreviation
                in ['idle', 'sr', 'sl', 'r', 'rl', 'su', 'sd', 'pin', 'spr', 'ft', 'p']]


class Preprocessing_parameters():
    def __init__(self, num_shifts: int, num_timesteps: int,
                 difference_mode: str = None, mediapipe_columns_for_diff: List[str] = None,
                 summands_pattern: List[int] = None, mediapipe_columns_for_sum: List[str] = None,
                 forearm_angle: bool = True):
        self.num_shifts = num_shifts
        self.num_timesteps = num_timesteps
        self.difference_mode = difference_mode
        self.mediapipe_columns_for_diff = mediapipe_columns_for_diff
        self.summands_pattern = summands_pattern
        self.mediapipe_columns_for_sum = mediapipe_columns_for_sum
        self.forearm_angle = forearm_angle

    def add_new_columns_to_column_lists(self, column_names: List[str]):
        # only add to cumsum list
 
        if any([column_name in self.mediapipe_columns_for_sum for column_name in column_names]):
            return None

        if not self.summands_pattern is None:
            self.mediapipe_columns_for_sum += column_names


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


def scale_to_body_size_and_dist_to_camera(vector_to_scale: np.array, window_df: pd.DataFrame):
    # vector_to_scale should be scaled by some uniform measure like the distance between hips and torso
    # print('window_df shape:', window_df.shape)
    # print('window_df', window_df)
    # print('window_df columns', window_df.columns)
    hip_mid_point = np.array([
        (window_df.iloc[0, window_df.columns.get_loc('left_hip_x')] + window_df.iloc[0, window_df.columns.get_loc('right_hip_x')]) / 2,
        (window_df.iloc[0, window_df.columns.get_loc('left_hip_y')] + window_df.iloc[0, window_df.columns.get_loc('right_hip_y')]) / 2
        ])
    shoulder_mid_point = np.array([
        (window_df.iloc[0, window_df.columns.get_loc('left_shoulder_x')] + window_df.iloc[0, window_df.columns.get_loc('right_shoulder_x')]) / 2,
        (window_df.iloc[0, window_df.columns.get_loc('left_shoulder_y')] + window_df.iloc[0, window_df.columns.get_loc('right_shoulder_y')]) / 2
    ])

    torso_length = np.linalg.norm(hip_mid_point - shoulder_mid_point)

    return (1 / torso_length) * vector_to_scale 


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

    # create a df with only the columns which should be considered in calculating the diff
    df_for_diff = extract_features(df, preproc_params.mediapipe_columns_for_diff)

    # number of samples in X
    num_samples = math.floor(
        (df_for_diff.shape[0] - num_timesteps + num_shifts) / num_shifts)

    data = df_for_diff.to_numpy()
    orig_columns = df_for_diff.columns

    if preproc_params.difference_mode == 'one':
        # diff between last and first of current sliding window

        num_features = data.shape[1]

        X = np.zeros((num_samples, num_features))

        for i in range(num_samples):
            X[i, :] = data[i * num_shifts + num_timesteps - 1, :] - \
                data[i * num_shifts, :]
            
            # apply scaling
            X[i, :] = scale_to_body_size_and_dist_to_camera(X[i, :], df.loc[i * num_shifts: i * num_shifts + num_timesteps - 1, :])

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

            # apply scaling
            X[i, :] = scale_to_body_size_and_dist_to_camera(X[i, :], df.loc[i * num_shifts: i * num_shifts + num_timesteps - 1, :])

        # creating the df
        column_diff_names = [orig_column +
                             "_diff" for orig_column in orig_columns]
        X_df = pd.DataFrame(data=X, columns=[
                            column_diff_name + f"_{str(i)}" for column_diff_name in column_diff_names for i in range(num_differences_in_a_sample)])

    return X, X_df


def correct_angle_boundary_diff(diff_np: np.array):
    # assumption: in two consecutive frames there can not be a angel diff of over np.pi in magnitude

    diff_np[np.where(diff_np > np.pi)[0]] = diff_np[np.where(diff_np > np.pi)[0]] - 2 * np.pi
    diff_np[np.where(diff_np < -np.pi)[0]] = diff_np[np.where(diff_np < -np.pi)[0]] + 2 * np.pi

    # if (np.abs(diff_np) > np.pi).any():
    #     print('here')

    return diff_np


def scale_angle_diff(angle_diff_np: np.array, window_start_index: int, num_timesteps: int, df, side: str):
    # print("angle_diff_np, window_start_index, num_timesteps, df, side", angle_diff_np, window_start_index, num_timesteps, df, side)
    if side == 'right':
        r = df.loc[window_start_index: window_start_index + num_timesteps - 1 , 'right_forearm_r'].to_numpy()
    elif side == 'left':
        r = df.loc[window_start_index: window_start_index + num_timesteps - 1 , 'left_forearm_r'].to_numpy()
    
    r_mid = (r[1:] + r[:-1]) / 2
    # print('r_mid', r_mid)
    r_mid_scaled = scale_to_body_size_and_dist_to_camera(r_mid*0.5, df.iloc[window_start_index: window_start_index + num_timesteps - 1, :])
    factor = np.zeros_like(r_mid_scaled)
    factor = np.power(r_mid_scaled, 6) * 100
    factor[np.where(r_mid_scaled > 1)] = r_mid_scaled[np.where(r_mid_scaled > 1)]
    transformed_angle_diff = angle_diff_np * factor
    # set a max value
    caped_angle_diff = np.min(np.c_[transformed_angle_diff, np.ones_like(transformed_angle_diff)*10], axis=1)
    caped_angle_diff = np.max(np.c_[caped_angle_diff, np.ones_like(caped_angle_diff)*-10], axis=1)

    # if (np.abs(caped_angle_diff) == 1000).any():
    #     print('here')

    return caped_angle_diff * 10


def calc_angle_diff(angle_np: np.array, window_start_index: int, num_timesteps: int, df: pd.DataFrame, side: str):
    angle_diff =  angle_np[window_start_index + 1: window_start_index + num_timesteps] - \
                angle_np[window_start_index: window_start_index + num_timesteps - 1]
    angle_diff = correct_angle_boundary_diff(angle_diff)
    return scale_angle_diff(angle_diff, window_start_index, num_timesteps, df, side)


def cumulative_sum(df: pd.DataFrame, preproc_params: Preprocessing_parameters, X) -> Tuple[np.array, pd.DataFrame]:
    # len(preproc_params.mediapipe_columns_for_sum)
    num_timesteps = preproc_params.num_timesteps
    num_shifts = preproc_params.num_shifts
    summands_pattern = preproc_params.summands_pattern

    # necessary as otherwise df is a view and the drop affects the original df
    df = df.copy()

    # create a df with only the columns which should be considered in calculating the diff
    df_for_sum = extract_features(df, preproc_params.mediapipe_columns_for_sum)
    
    angle_involved = True
    if 'right_forearm_angle' in df_for_sum.columns:
        right_forearm_angle = df_for_sum['right_forearm_angle'].to_numpy()
        left_forearm_angle = df_for_sum['left_forearm_angle'].to_numpy()
        df_for_sum.drop(['right_forearm_angle', 'left_forearm_angle'], axis=1, inplace=True)

    # number of samples in X
    num_samples = math.floor(
        (df_for_sum.shape[0] - num_timesteps + num_shifts) / num_shifts)

    data = df_for_sum.to_numpy()
    
    num_angle_features = 0

    orig_columns = list(df_for_sum.columns)
    # add the angle column names at the end
    if angle_involved:
        orig_columns += ['right_forearm_angle', 'left_forearm_angle']
        num_angle_features = 2


    num_features = (data.shape[1] + num_angle_features) * sum(summands_pattern)

    # X = np.zeros((num_samples, num_features))

    # print("num_samples, num_features", num_samples, num_features)

    for i in range(num_samples):
        window_start_index = i * num_shifts
        window_np = data[window_start_index: window_start_index + num_timesteps, :]
        diff_window_np = window_np[1:, :] - window_np[:-1, :]

        # apply scaling to diff here, as the angle should not be scaled
        # diff_window_np = scale_to_body_size_and_dist_to_camera(diff_window_np, df.loc[i * num_shifts: i * num_shifts + num_timesteps - 1, :])

        if angle_involved:
            right_forearm_angle_diff = calc_angle_diff(right_forearm_angle, window_start_index, num_timesteps, df, side='right')
            left_forearm_angle_diff = calc_angle_diff(left_forearm_angle, window_start_index, num_timesteps, df, side='left')
        
            diff_window_np = np.c_[diff_window_np, right_forearm_angle_diff, left_forearm_angle_diff]

        # calc the cumulative sum over the whole window
        cumsum = np.cumsum(diff_window_np, axis=0)

        # extract only the cumsums specified in the pattern
        cumsum_features = cumsum[np.array(summands_pattern, dtype=bool), :]

        # adding the features to X
        X[i, :-2] = cumsum_features.flatten('F').reshape(1, -1)

        # apply scaling to diff here, as the angle is now multiplied by projected wrist to elbow dist, which needs to be scaled
        X[i, :-2] = scale_to_body_size_and_dist_to_camera(X[i, :-2], df.loc[i * num_shifts: i * num_shifts + num_timesteps - 1, :])

    # # creating the df
    # column_cumsum_names = [orig_column +
    #                         "_cumsum" for orig_column in orig_columns]
    # X_df = pd.DataFrame(data=X, columns=[
    #                     column_cumsum_name + f"_{str(i)}" for column_cumsum_name in column_cumsum_names for i in range(sum(summands_pattern))])

    # return X, X_df


def calc_forearm_angle(df: pd.DataFrame):
    # copy otherwise original df will be changed
    df = df.copy()

    # looking at the angle values might be confusing, as the y coordinate is quasi flipped

    right_wrist_x = df['right_wrist_x'].to_numpy()
    right_wrist_y = df['right_wrist_y'].to_numpy()

    right_elbow_x = df['right_elbow_x'].to_numpy()
    right_elbow_y = df['right_elbow_y'].to_numpy()

    forearm_vector_x = right_wrist_x - right_elbow_x
    forearm_vector_y = right_wrist_y - right_elbow_y

    # calc angle with only positive angles
    angle_np = np.arctan2( forearm_vector_y, forearm_vector_x )
    angle_np[np.where(angle_np < 0)[0]] = angle_np[np.where(angle_np < 0)[0]] + 2 * np.pi

    r = np.sqrt(np.power(forearm_vector_x, 2) + np.power(forearm_vector_y, 2))

    df['right_forearm_angle'] = angle_np
    df['right_forearm_r'] = r

    left_wrist_x = df['left_wrist_x'].to_numpy()
    left_wrist_y = df['left_wrist_y'].to_numpy()

    left_elbow_x = df['left_elbow_x'].to_numpy()
    left_elbow_y = df['left_elbow_y'].to_numpy()

    forearm_vector_x = left_wrist_x - left_elbow_x
    forearm_vector_y = left_wrist_y - left_elbow_y

    # calc angle with only positive angles
    angle_np = np.arctan2( forearm_vector_y, forearm_vector_x )
    angle_np[np.where(angle_np < 0)[0]] = angle_np[np.where(angle_np < 0)[0]] + 2 * np.pi

    r = np.sqrt(np.power(forearm_vector_x, 2) + np.power(forearm_vector_y, 2))
    
    df['left_forearm_angle'] = angle_np
    df['left_forearm_r'] = r

    return df


def shoulder_wrist_difference(df: pd.DataFrame, preproc_params: Preprocessing_parameters, X):

    num_timesteps = preproc_params.num_timesteps
    num_shifts = preproc_params.num_shifts
    num_samples = math.floor(
        (df.shape[0] - num_timesteps + num_shifts) / num_shifts)

    shoulder_x = df[["right_shoulder_x", "left_shoulder_x"]].to_numpy()
    shoulder_y = df[["right_shoulder_y", "left_shoulder_y"]].to_numpy()
    elbow_x = df[["right_elbow_x", "left_elbow_x"]].to_numpy()
    elbow_y = df[["right_elbow_y", "left_elbow_y"]].to_numpy()
    wrist_x = df[["right_wrist_x", "left_wrist_x"]].to_numpy()
    wrist_y = df[["right_wrist_y", "left_wrist_y"]].to_numpy()

    # Wähle den nten Frame in jedem Fenster zur Differenzberechnung
    n = num_timesteps
    select_position = n - 1
    for i in range(num_samples):
        selection_index = i * num_shifts + select_position

        x_shoulder_elbow_diff = np.subtract(shoulder_x[selection_index], elbow_x[selection_index])
        x_elbow_wrist_diff = np.subtract(elbow_x[selection_index], wrist_x[selection_index])

        y_shoulder_elbow_diff = np.subtract(shoulder_y[selection_index], elbow_y[selection_index])
        y_elbow_wrist_diff = np.subtract(elbow_y[selection_index], wrist_y[selection_index])

        right_dist = np.sqrt(np.power(x_shoulder_elbow_diff[0], 2) + np.power(y_shoulder_elbow_diff[0], 2)) + \
                     np.sqrt(np.power(x_elbow_wrist_diff[0], 2) + np.power(y_elbow_wrist_diff[0], 2))

        left_dist = np.sqrt(np.power(x_shoulder_elbow_diff[1], 2) + np.power(y_shoulder_elbow_diff[1], 2)) + \
                    np.sqrt(np.power(x_elbow_wrist_diff[1], 2) + np.power(y_elbow_wrist_diff[1], 2))

        X[i, -2] = right_dist
        X[i, -1] = left_dist

        X[i, -2:] = scale_to_body_size_and_dist_to_camera(X[i, -2:], df.loc[selection_index: selection_index, :])



def determine_label_from_ground_truth_vector(ground_truth_df: pd.DataFrame, num_timesteps: int,
                                             num_shifts: int) -> np.array:
    num_samples = math.floor(
        (ground_truth_df.shape[0] - num_timesteps + num_shifts) / num_shifts)

    y = np.zeros((num_samples, ground_truth_df.shape[1]))

    data = ground_truth_df.to_numpy()

    idle_hone_hot = np.zeros_like(data[0*num_shifts + num_timesteps-6, :])
    idle_hone_hot[Labels.idle.value] = 1
    rotate_one_hot = np.zeros_like(data[0*num_shifts + num_timesteps-6, :])
    rotate_one_hot[Labels.rotate.value] = 1
    rotate_left_one_hot = None
    if 'rotate_left' in Labels.get_label_list():
        rotate_left_one_hot = np.zeros_like(data[0*num_shifts + num_timesteps-6, :])
        rotate_left_one_hot[Labels.rotate_left.value] = 1

    for i in range(num_samples):

        middle_one_hot = data[i*num_shifts + num_timesteps-(int(np.ceil(num_timesteps/2)) + 1), :]
        second_to_last_one_hot = data[i*num_shifts + num_timesteps-2, :] # vom vorletzten im window ablesen
        last_one_hot = data[i*num_shifts + num_timesteps-1, :] # vom letzten im window ablesen

        if (last_one_hot == rotate_one_hot).all() and (middle_one_hot == rotate_one_hot).all():
            # both -> rotate
            y[i, :] = rotate_one_hot
        else:
            if (second_to_last_one_hot == rotate_one_hot).all():
                # not both, but second to last -> idle
                y[i, :] = idle_hone_hot
            else:
                # not both and not even second to last -> check for rotate left
                if rotate_left_one_hot is None:
                    y[i, :] = second_to_last_one_hot
                else:
                    if (last_one_hot == rotate_left_one_hot).all() and (middle_one_hot == rotate_left_one_hot).all():
                        # both -> rotate
                        y[i, :] = rotate_left_one_hot
                    else:
                        if (second_to_last_one_hot == rotate_left_one_hot).all():
                            # not both, but second to last -> idle
                            y[i, :] = idle_hone_hot
                        else:
                            # not rotate related -> default 
                            y[i, :] = second_to_last_one_hot

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


def create_X(df: pd.DataFrame, preproc_params: Preprocessing_parameters, verbose=False) -> Tuple[np.array, pd.DataFrame]:
    # num_shifts: int, num_timesteps: int, difference_mode: str, summands_pattern: List[int]) -> Tuple[np.array, pd.DataFrame]:
    # evtl add fuctions for other features here

    # number of samples in X
    num_samples = math.floor(
        (df.shape[0] - preproc_params.num_timesteps + preproc_params.num_shifts) / preproc_params.num_shifts)


    # note that forearm_angle is different to differences and cumsum, as is on coordinate level and needs to go through diff or cumsum
    if preproc_params.forearm_angle:
        # t3 = time.perf_counter()
        df = calc_forearm_angle(df)
        # print('calc_forearm_angle time', time.perf_counter() - t3)

        # add the angle names to the lists so that diff and cumsum is also applied to it
        preproc_params.add_new_columns_to_column_lists(
            ['right_forearm_angle', 'left_forearm_angle', 'right_forearm_r', 'left_forearm_r'])
    
    orig_columns = preproc_params.mediapipe_columns_for_sum

    # trying to initialize X at the start to avoid having to to concatenation, which is quite slow for live mode performance
    # df.shape[1] columns to cumsum features and 2 features for wrist shoulder vector
    num_features = len(preproc_params.mediapipe_columns_for_sum) * sum(preproc_params.summands_pattern) + 2

    X = np.zeros((num_samples, num_features))

    

    if preproc_params.difference_mode:
        print(f'{bcolors.FAIL}WARNING: difference mode is deactivated, needs refactoring{bcolors.ENDC}')
    #     X_diff, X_diff_df = calc_differences(df, preproc_params)
    # else:
    #     X_diff, X_diff_df = np.array([]).reshape(num_samples, 0), None

    if preproc_params.summands_pattern:
        # t4 = time.perf_counter()
        # write the cumsum in the first columns of X
        cumulative_sum(df, preproc_params, X)
        # print('cumsum calc time', time.perf_counter() - t4)
    else:
        raise RuntimeError('summands_pattern needed, implementation for diff needed')

    # add shoulder wrist dist in the last 2 columns of X
    shoulder_wrist_difference(df, preproc_params, X)

    # t6 = time.perf_counter()

    X = X.round(6)

    # creating the df
    column_cumsum_names = [orig_column +
                            "_cumsum" for orig_column in orig_columns]
    cumsum_column_names = [column_cumsum_name + f"_{str(i)}" for column_cumsum_name in column_cumsum_names for i in range(sum(preproc_params.summands_pattern))]
    shoulder_wrist_column_names = ["right_shoulder_elbow_wrist_dist", "left_shoulder_elbow_wrist_dist"]
    X_df = pd.DataFrame(data=X, columns=cumsum_column_names + shoulder_wrist_column_names)


    # print('create x inner time', time.perf_counter() - t6)

    return X, X_df


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

# train_val_test /in {'train', 'val', 'test'}
def handle_preprocessing(labeled_frames_folder_path: Path, preprocessed_frames_folder_path: Path, preproc_params: Preprocessing_parameters,
                         only_optional_bool: bool, train_val_test: str = 'train'):
    """gets all '*_labeled.csv' files under the specified folder, does preprocessing with the specified parameters and saves it in the other specified folder.
    Args:
        labeled_frames_folder_path (Path): load folder topath
        preprocessed_frames_folder_path (Path): to folder path
    """
    err_files = []
    search_ending = '**/*_' + train_val_test + '_labeled.csv'
    for labeled_csv_file_path in tqdm(labeled_frames_folder_path.glob(search_ending)):
        print('Now on file: ', labeled_csv_file_path)
        # try:
        if only_optional_bool:
            if 'mandatory' in str(labeled_csv_file_path):
                continue
        if 'nina' in str(labeled_csv_file_path):
            continue
        _, nn_input_df = preprocessing(labeled_csv_file_path, preproc_params)

        nn_input_df.to_csv(preprocessed_frames_folder_path /
                            labeled_csv_file_path.name.replace("_labeled.csv", "_preproc.csv"))

        # except Exception as e:
        #     print("error in file", labeled_csv_file_path, e)
        #     err_files.append((labeled_csv_file_path, e))
    
    print('error in files: \n', err_files)



if __name__ == '__main__':
    # FILE_PATH = r'data\labeled_frames\demo_video_csv_with_ground_truth_rotate_labeled.csv'
    # nn_input, nn_input_df = preprocessing(FILE_PATH, mediapipe_colums_for_diff.copy(),
    #                                       num_shifts=1, num_timesteps=4, difference_mode='every')
    # nn_input_df.to_csv('nn_input_test.csv')

    # preproc_params = Preprocessing_parameters(
    #     num_shifts=1, num_timesteps=10,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
    #     summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)
    # preproc_params = Preprocessing_parameters(
    #     num_shifts=1, num_timesteps=10,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
    #     summands_pattern=[1, 0, 1, 0, 1, 0, 1, 0, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)
    #preproc_params = Preprocessing_parameters(
     #   num_shifts=1, num_timesteps=8,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
     #   summands_pattern=[1, 0, 1, 0, 1, 0, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)
    preproc_params = Preprocessing_parameters(
         num_shifts=1, num_timesteps=10,  # difference_mode='one', mediapipe_columns_for_diff= mediapipe_colums_for_diff,
         summands_pattern=[1, 1, 1, 1, 1, 1, 1, 1, 1], mediapipe_columns_for_sum=mediapipe_columns_for_sum)

    
    Labels = LabelsOptional

    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), Path(
    #     r'../../data\preprocessed_frames\final\train\optional'), preproc_params, train_val_test='train')

    # 10 cumsum all
    handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
                         Path( r'../../data\preprocessed_frames\new_window=10,cumsum=all\train\mandatory_data'), 
                         preproc_params, 
                         only_optional_bool=False, 
                         train_val_test='train') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/rotate_right/train/03-19_nina_rotate_train_labeled.csv'), KeyError(nan))]
    
    handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
                         Path( r'../../data\preprocessed_frames\new_window=10,cumsum=all\validation\mandatory_data'), 
                         preproc_params, 
                         only_optional_bool=False, 
                         train_val_test='val') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/swipe_right/validation/03-19_nina_swipe_right_val_labeled.csv'), KeyError(nan))]
    
    handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
                         Path( r'../../data\preprocessed_frames\new_window=10,cumsum=all\validation\optional'), 
                         preproc_params, 
                         only_optional_bool=True, 
                         train_val_test='val') #[(WindowsPath('../../data/labeled_frames/ready_to_train/pinch/validation/03-19_nina_pinch_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/point/val/03-19_nina_point_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/validation/03-19_nina_swipe_up_val_labeled.csv'), KeyError(nan))]
    
    handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
                         Path( r'../../data\preprocessed_frames\new_window=10,cumsum=all\train\optional'), 
                         preproc_params, 
                         only_optional_bool=True, 
                         train_val_test='train') #[(WindowsPath('../../data/labeled_frames/ready_to_train/spread/train/03-19_nina_spread_train_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/train/03-19_nina_swipe_up_train_labeled.csv'), KeyError(nan))]


    # 10 cumsum every_second
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'),
    #                       Path( r'../../data\preprocessed_frames\new_window=10,cumsum=every_second\train\mandatory_data'),
    #                       preproc_params,
    #                       only_optional_bool=False,
    #                       train_val_test='train') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/rotate_right/train/03-19_nina_rotate_train_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'),
    #                       Path( r'../../data\preprocessed_frames\new_window=10,cumsum=every_second\validation\mandatory_data'),
    #                       preproc_params,
    #                       only_optional_bool=False,
    #                       train_val_test='val') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/swipe_right/validation/03-19_nina_swipe_right_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'),
    #                       Path( r'../../data\preprocessed_frames\new_window=10,cumsum=every_second\validation\optional'),
    #                       preproc_params,
    #                       only_optional_bool=True,
    #                       train_val_test='val') #[(WindowsPath('../../data/labeled_frames/ready_to_train/pinch/validation/03-19_nina_pinch_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/point/val/03-19_nina_point_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/validation/03-19_nina_swipe_up_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'),
    #                       Path( r'../../data\preprocessed_frames\new_window=10,cumsum=every_second\train\optional'),
    #                       preproc_params,
    #                       only_optional_bool=True,
    #                       train_val_test='train')

    # 8 cumsum all
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=all\train\mandatory_data'), 
    #                      preproc_params, 
    #                      only_optional_bool=False, 
    #                      train_val_test='train') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/rotate_right/train/03-19_nina_rotate_train_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=all\validation\mandatory_data'), 
    #                      preproc_params, 
    #                      only_optional_bool=False, 
    #                      train_val_test='val') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/swipe_right/validation/03-19_nina_swipe_right_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=all\validation\optional'), 
    #                      preproc_params, 
    #                      only_optional_bool=True, 
    #                      train_val_test='val') #[(WindowsPath('../../data/labeled_frames/ready_to_train/pinch/validation/03-19_nina_pinch_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/point/val/03-19_nina_point_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/validation/03-19_nina_swipe_up_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=all\train\optional'), 
    #                      preproc_params, 
    #                      only_optional_bool=True, 
    #                      train_val_test='train') #[(WindowsPath('../../data/labeled_frames/ready_to_train/spread/train/03-19_nina_spread_train_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/train/03-19_nina_swipe_up_train_labeled.csv'), KeyError(nan))]


    # 8 cumsum every_second
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=every_second\train\mandatory_data'), 
    #                      preproc_params, 
    #                      only_optional_bool=False, 
    #                      train_val_test='train') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/rotate_right/train/03-19_nina_rotate_train_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train\mandatory_gestures'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=every_second\validation\mandatory_data'), 
    #                      preproc_params, 
    #                      only_optional_bool=False, 
    #                      train_val_test='val') # [(WindowsPath('../../data/labeled_frames/ready_to_train/mandatory_gestures/swipe_right/validation/03-19_nina_swipe_right_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=every_second\validation\optional'), 
    #                      preproc_params, 
    #                      only_optional_bool=True, 
    #                      train_val_test='val') #[(WindowsPath('../../data/labeled_frames/ready_to_train/pinch/validation/03-19_nina_pinch_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/point/val/03-19_nina_point_val_labeled.csv'), KeyError(nan)), (WindowsPath('../../data/labeled_frames/ready_to_train/swipe_up/validation/03-19_nina_swipe_up_val_labeled.csv'), KeyError(nan))]
    
    # handle_preprocessing(Path(r'../../data\labeled_frames\ready_to_train'), 
    #                      Path( r'../../data\preprocessed_frames\window=8,cumsum=every_second\train\optional'), 
    #                      preproc_params, 
    #                      only_optional_bool=True, 
    #                      train_val_test='train')



    print('done')
