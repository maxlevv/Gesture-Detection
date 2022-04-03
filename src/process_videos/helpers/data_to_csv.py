import pandas as pd
import yaml


class CSVDataWriter:
    def __init__(self):
        self.frame_list = []
        self.timestamps = []
        self.column_names = self.load_keypoint_mapping_from_file(r"C:\Users\Jochen\Jonas\ML\ml_dev_repo\src\process_videos\keypoint_mapping.yml")

    def read_data(self, data, timestamp):
        frame = []
        if data is None:
            return None
        for i in range(33):
            frame.append(data.landmark[i].x)
            frame.append(data.landmark[i].y)
            frame.append(data.landmark[i].z)
            frame.append(data.landmark[i].visibility)
        self.frame_list.append(frame)
        self.timestamps.append(timestamp)
        return frame

    def to_csv(self, output_path):
        frames = pd.DataFrame(self.frame_list, columns=self.column_names, index=self.timestamps)
        print(frames)
        frames.index.name = "timestamp"
        frames.index = frames.index.astype(int)
        print(frames)
        frames.round(5).to_csv(output_path)
        print(frames.round(5))

    def load_keypoint_mapping_from_file(self, file):
        with open(file, "r") as yaml_file:
            mappings = yaml.safe_load(yaml_file)
            KEYPOINT_NAMES = mappings["face"]
            KEYPOINT_NAMES += mappings["body"]
        return ["%s_%s" % (joint_name, jdn) for joint_name in KEYPOINT_NAMES for jdn in
                ["x", "y", "z", "confidence"]]
