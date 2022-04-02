from pathlib import Path
import cv2
import mediapipe as mp
from helpers import data_to_csv as dtc
import time

def video_to_csv(video_folder_path: Path, raw_frames_folder_path: Path, flip_image: bool):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    show_video = False

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    for video_file_path in video_folder_path.glob('*.mp4'):
        cap = cv2.VideoCapture(str(video_file_path))
        # cap = cv2.VideoCapture(index=0)

        result_csv_filename = raw_frames_folder_path / f"{video_file_path.name.replace('.mp4', '_')}{current_time}_raw.csv"

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

                if flip_image:
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                image.flags.writeable = False
                results = pose.process(image)

                if show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
                csv_writer.read_data(data=results.pose_landmarks, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC))
        
        csv_writer.to_csv(str(result_csv_filename))
        cap.release()


def testing():
    video_folder_path = Path(r'C:\Users\Max\Documents\Master Würzburg\Machine Learning\final-project-getting-started\demo_data')
    raw_frames_folder_path = Path(r'..\..\data\raw_frames')
    video_to_csv(video_folder_path, raw_frames_folder_path, flip_image=True)


def converting():
    #video_folder_path = Path(r'C:\Users\Max\PycharmProjects\ml_dev_repo\videos\rotate_right')
    #video_folder_path = Path(r'C:\Users\Max\PycharmProjects\ml_dev_repo\videos\swipe_right')
    #video_folder_path = Path(r'C:\Users\Sepp\Videos\ml_projekt_test')
    video_folder_path = Path(r'C:\Users\hornh\Documents\ml_projekt_videos\val_vids_tamara')
    #raw_frames_folder_path = Path(r'..\..\data\raw_frames\rotate_right')
    #raw_frames_folder_path = Path(r'..\..\data\raw_frames\swipe_right')
    raw_frames_folder_path = Path(r'C:\Users\hornh\OneDrive\Dokumente\Uni\Info\MachineLearning\project_dev_repo\ml_dev_repo\data\raw_frames\tamara_val')
    video_to_csv(video_folder_path, raw_frames_folder_path, flip_image=False)


if __name__ == '__main__':
    converting()