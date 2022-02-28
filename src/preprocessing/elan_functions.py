import pandas as pd
from pathlib import Path


def find_corresponding_raw_frame_to_elan_annot(raw_frames_folder_path: Path, common_datum_identifier: str):
    possible_raw_frame_paths = list(
        raw_frames_folder_path.glob(f'**/{common_datum_identifier}*_raw.csv'))
    if len(possible_raw_frame_paths) == 0:
        raise RuntimeError(
            f'No possible raw frame found with name {common_datum_identifier}')
    elif len(possible_raw_frame_paths) > 1:
        raise RuntimeError(
            f'Multiple possible raw frames found: \n{possible_raw_frame_paths}')
    else:
        return possible_raw_frame_paths[0]


def add_elan_labels_to_frame(elan_annotation_folder_path: Path, raw_frames_folder_path: Path, labeled_frames_folder_path: Path):
    """goes through each '*_annot.txt' file in elan folder and searches a corresponding raw csv under the given path, adds the labels and saves the df and
    """
    for elan_annoation_file_path in elan_annotation_folder_path.glob('**/*annot.txt'):
        common_datum_identifier = elan_annoation_file_path.name.replace(
            '_annot.txt', '')
        raw_frame_file_path = find_corresponding_raw_frame_to_elan_annot(
            raw_frames_folder_path, common_datum_identifier)

        # sep needed, as in vs code csv extensions whitespaces are added which should not be included in the data    
        df = pd.read_csv(raw_frame_file_path, sep=' *,', engine='python') 

        # set timestamp column as index, so we can do timerange based selections
        df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df.index = df.index.rename("timestamp")

        # the default label will be 'idle'
        df["ground_truth"] = "idle"

        # read the ELAN file as CSV; only read the relevant columns (specified by `usecols`)
        annotations = pd.read_csv(elan_annoation_file_path, sep="\t", header=None, usecols=[
                                  3, 5, 8], names=["start", "end", "label"])

        # convert start and end timestamps into a datetime-datatype – this is important for the next step
        annotations["start"] = pd.to_timedelta(annotations["start"], unit="s")
        annotations["end"] = pd.to_timedelta(annotations["end"], unit="s")

        # iterate over each entry from the ELAN file, select the corresponding df and label them accordingly:
        for idx, ann in annotations.iterrows():
            annotated_df = (df.index >= ann["start"]) & (
                df.index <= ann["end"])
            df.loc[annotated_df, "ground_truth"] = ann["label"]

        # convert index back to integers, so it looks nicer in CSV
        df.index = df.index.astype(int) // 1_000_000

        # save all data with the new ground_truth column into a new file:
        df.to_csv(labeled_frames_folder_path /
                  (common_datum_identifier + "_labeled.csv"), index=True)


def test():
    elan_annotation_folder_path = Path(r'data\elan_annotations')
    raw_frames_folder_path = Path(r'data\raw_frames')
    labeled_frames_folder_path = Path(r'data\labeled_frames')

    add_elan_labels_to_frame(elan_annotation_folder_path,
                             raw_frames_folder_path, labeled_frames_folder_path)


def do():
    elan_annotation_folder_path = Path(r'C:\Users\hornh\Documents\ml_projekt_videos\at_home\rotate_right')
    elan_annotation_folder_path = Path(r'C:\Users\hornh\Documents\ml_projekt_videos\at_home\swipe_left')
    elan_annotation_folder_path = Path(r'C:\Users\hornh\Documents\ml_projekt_videos\at_home\swipe_right')
    elan_annotation_folder_path = Path(r'data\elan_annotations')
    raw_frames_folder_path = Path(r'data\raw_frames')
    labeled_frames_folder_path = Path(r'data\labeled_frames\rotate_right')
    labeled_frames_folder_path = Path(r'data\labeled_frames\swipe_left')
    labeled_frames_folder_path = Path(r'data\labeled_frames\swipe_right')
    labeled_frames_folder_path = Path(r'data\labeled_frames')

    add_elan_labels_to_frame(elan_annotation_folder_path,
                             raw_frames_folder_path, labeled_frames_folder_path)



if __name__ == '__main__':
    do()
