from pathlib import Path
import cv2
import mediapipe as mp
from helpers import data_to_csv as dtc
import time

def video_to_csv(video_folder_path: Path, raw_frames_folder_path: Path):
    mp_pose = mp.solutions.pose

    # current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    for video_file_path in video_folder_path.glob('*.mp4'):
        cap = cv2.VideoCapture(str(video_file_path))

        result_csv_filename = raw_frames_folder_path / f"{video_file_path.name.replace('.mp4', '_')}_raw.csv"

        csv_writer = dtc.CSVDataWriter()
        frame_count = 0
        success = True
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened() and success:
                if frame_count % 100 == 0: print(f'On frame {frame_count}')
                frame_count += 1

                success, image = cap.read()
                if not success:
                    break
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
            
                csv_writer.read_data(data=results.pose_landmarks, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC))
        
        csv_writer.to_csv(str(result_csv_filename))
        cap.release()


if __name__ == '__main__':
    video_folder_path = Path(r'data\video_files')
    raw_frames_folder_path = Path(r'data\raw_frames')
    video_to_csv(video_folder_path, raw_frames_folder_path)