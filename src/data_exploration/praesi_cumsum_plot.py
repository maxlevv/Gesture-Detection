import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

preproc_file_path = Path(r'data\preprocessed_frames\final\train\mandatory_data\02-24_jonas_swipe_left_train_preproc.csv')
df = pd.read_csv(preproc_file_path, delimiter=' *,', engine='python')
print(df.columns)

fig, ax = plt.subplots(figsize=(40, 17))
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_wrist_x_cumsum_5'], where='mid', color='blue', label="Cumsum für x-coord. re. Handgelenk")
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_forearm_angle_cumsum_5'], where='mid', color='green', label="Cumsum für Winkel re. Unterarm")
ax.step(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'right_forearm_r_cumsum_5'], where='mid', color='red', label="Cumsum für right_forearm_r")
ax.plot(df.index[60:60 + 180], df.loc[60:60 + 180-1, 'gt_sl'], label='Ground truth (swipe left)', color='orange')

ax.set_xlabel("Nummer des Frames", fontsize=30, labelpad=30)
ax.set_ylabel("Kumulative Summe einer Körperpunkt-Koord. (Cumsum)\nbzw. Ground Truth", fontsize=30, labelpad=30)
ax.set_title("Gestenspezifische Features für die Geste Swipe-Left", fontsize=40, pad=35)

ax.legend(loc='best', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=25)

# plt.show()
fig.savefig('nina_new_praesi_plot.png')