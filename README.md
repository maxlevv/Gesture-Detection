# Gesture Detection - Machine Learning Team 16 Repository

Respository for the final project of the Machine Learning course

## Requirements for this repository
- python version 3.7.9 
- packages: requirments.txt


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
  

