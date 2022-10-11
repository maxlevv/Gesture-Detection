import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

preproc_file_path = Path('../../data/preprocessed_frames/final/train/mandatory_data/02-24_jonas_swipe_left_train_preproc.csv')
df = pd.read_csv(preproc_file_path, delimiter=' *,', engine='python')
print(df.columns)

#fig, axs = plt.subplots(6, 1, figsize=(80, 50))
fig, ax = plt.subplots(figsize=(18, 5))
#for i in range(6):
#    axs[i].step(df.index[i*200:i*200 + 200], df.loc[i*200:i*200 + 200-1, 'right_wrist_x'], where='mid', color='blue')
#    #axs[i].step(df.index[i*200:i*200 + 200], df.loc[i*200:i*200 + 200-1, 'right_wrist_y'], where='mid', color='green')
#    axs[i].step(df.index[i*200:i*200 + 200], df.loc[i*200:i*200 + 200-1, 'left_wrist_x'], where='mid', color='yellow')
#    axs[i].plot(df.index[i*200:i*200 + 200], df.loc[i*200:i*200 + 200-1, 'ground_truth'])
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_wrist_x_cumsum_5'], where='mid', color='blue', label="right_wrist_x_cumsum_8")
#ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_wrist_y_cumsum_8'], where='mid', color='green', label="right_wrist_y_cumsum_8")
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_forearm_angle_cumsum_5'], where='mid', color='green', label="right_forearm_angle_cumsum_8")
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_forearm_r_cumsum_5'], where='mid', color='red', label="right_forearm_r_cumsum_8")
#ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_forearm_angle_cumsum_0'], where='mid', color='black', label="right_wrist_y_cumsum_8")
ax.plot(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'gt_r'])

ax.set_xlabel("Nummer des Frames", fontsize=30, labelpad=30)
ax.set_ylabel("Kumulative Summe", fontsize=30, labelpad=30)
ax.set_title("Kumulative Summe der Zwischen-Frame-Differenzen ausgewählter Features für die Geste Swipe-Left", fontsize=40, pad=35)

ax.legend(loc='best', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)

plt.show()
fig.savefig('nina_new_praesi_plot.png')