import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "../../data/preprocessed_frames/demo_video_csv_with_ground_truth_rotate.csv"
frames_all = pd.read_csv(FILE_PATH)

def extract_features(frames, features: list):
    features.append("ground_truth")
    return frames[features]

def preprocessing_difference(frames, number_timestamps: int, number_shifts: int):
    m = int(frames.shape[0] / number_shifts - number_timestamps)
    n = frames.shape[1] - 1

    # data matrix to be returned
    X = np.array(np.zeros((m, n)))

    # ground truth vector to be returned, for now m x 1 (only rotate or not)
    y = np.array(np.zeros(m))

    for i in range(m):
        data = ((frames.iloc[:, :-1]).iloc[i * number_shifts : (i * number_shifts) + number_timestamps]).to_numpy()
        difference = data[number_timestamps - 1, :] - data[0]
        X[i,:] = difference

        if frames["ground_truth"].iloc[(i * number_shifts) + number_timestamps] != "idle":
            y[i] = 1

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(X[0:, 1])
    ax.plot(X[0:, 0])
    ax.plot(y / 6)
    plt.show()
    return X, y

frames_rotate = extract_features(frames_all, ["left_index_x", "left_index_y"])

timestamps = 5
shifts = 5
X, y = preprocessing_difference(frames_rotate, timestamps, shifts)

#X2, y2 = preprocessing_difference(frames_rotate, 3, 3)
#data = np.concatenate((X, X2), axis=0)
#print(data.shape[0])
