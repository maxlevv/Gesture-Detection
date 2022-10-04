# Gesture Detection - Machine Learning Team 16 Repository

Respository for the final project of the Machine Learning course

## Requirements for this repository
- python version 3.7.9 
- packages: requirements.txt

## Setup 
1. Install python version 3.7.9
2. Install pipenv by using 
     - `pip install virtualenv`
3. Create a new virtual environment at your disired location with 
    - `virtualenv _name_of_your_env_ --python=python3.7.9` 
4. Acivate the environment and install dependencies
   -  `pip install -r requirements.txt`

## How to run the two Evaluation Modes
1. **Test mode**

    In order to run the test mode for grading the performance requirements, switch to the directory `src/performance_score/` and run the script `gesture_detection.py` with 
    `python gesture_detection.py`.

2. **Prediction mode**

    In order to run the prediction mode for controlling the slideshow, switch to the directory `src/slideshow/` and run the script `gesture_detection_for_live_mode.py` with `python gesture_detection_for_live_mode.py`.


## Structure
- data : 
  - raw frames : csv files processed by mediapipe, raw mediapipe features
  - elan_annotations: txt files with annotations from ELAN labeling tool
  - labeled frames : labeled csv files with annotation txt files produced by ELAN
  - preprocessed frames : preprocessed csv files

  
&nbsp;

- src :
  - preprocessing : scripts and notebooks for data exploration and functions for preprocessing the data so that it is ready to be put into the model
  - modeling : functions for applying the network with different architectures and hyperparameters
  - evaluation : functions for evaluating trained models (statistics, evtl. visializations)
  - slideshow : everything slide show related and especially gesture_detection_for_live_mode.py
  - data_exploration: exploration of mediapipe features
  - process_videos: code for converting video to csv with mediapipe and threaded camera implementation
  - performance_score: functionality for calculating the performance score in the exam
  
&nbsp;

- saved_runs : containing folders for groups of runs containing folders for individual runs containing meta_data.json files and .npy files of trained weights
  
&nbsp;

- archive: code not needed any more, but might be somewhat interesting  


## Runs and Plots used for the Presentation

- The data that we used to generate the plots in the presentation can be found in saved_runs/Visualized_Runs
- To create the plots we used the functions *generate_mean_f1_overview_plot*, *evaluate_runs* and *test_eval* from the file src/evaluation/evaluate.py 
- To visualize the MediaPipe data and our preprocessed data we used the notebook src/data_exploration/feature_visualization.ipynb 
