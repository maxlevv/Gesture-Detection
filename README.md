# Gesture Detection - Machine Learning Team 16 Repository

Respository for the final project of the Machine Learning course

## Requirements for this repository
- python version 3.7.9 
- packages: requirements.txt

## Setup 
1. Open your terminal
2. Switch to this directory
3. Install dependencies
   - Run `pip install -r requirements.txt`, this installs the required packages.

## How to run the two Evaluation Modes
1. Test mode

In order to run the test mode for grading the performance requirements, switch to the directory `src/performance_score/` and run the script `gesture_detection.py` with `python gesture_detection.py`.

2. Prediction mode 

In order to run the prediction mode for controlling the slideshow, switch to the directory `src/slideshow/` and run the script `gesture_detection_for_live_mode.py` with `python gesture_detection_for_live_mode.py`.


## Structure
- data : 
  - raw frames : csv files processed by mediapipe raw
  - labeled frames : labeled csv files with annotation txt files produced by ELAN
  - preprocessed frames : preprocessed csv files
  
&nbsp;

- src :
  - preprocessing : scripts and notebooks for data exploration and functions for preprocessing the data so that it is ready to be put into the model
  - modeling : neural network and functions for it; functions for applying the network with different architectures and hyperparameters
  - evalutaion : functions for evaluating trained models (statistics, evtl. visializations)
  - slideshow : everything slide show related
  - pipline? (probably not needed)
  
&nbsp;

- saved_runs : containing folders for groups of runs containing folders for individual runs containing meta_data.json files and .npy files of trained weights
  
&nbsp;

- archive: code not needed any more

