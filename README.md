# ml_dev_repo

Project for developing the final project

## Requirements
- python version 3.7.9 (this one works with mediapipe others may not)

## Structure
- data : (for larg training sets later on add this folder to gitignore and store only locally and maybe on some cloud platform)
  - raw frames : csv files processed by mediapipe raw
  - labeled frames : labeled csv files with annotation txt files produced by ELAN
  - preprocessed frames : preprocessed csv files

- src :
  - preprocessing : scripts and notebooks for data exploration and functions for preprocessing the data so that it is ready to be put into the model
  - modeling : neural network and functions for it; functions for applying the network with different architectures and hyperparameters
  - evalutaion : functions for evaluating trained models (statistics, evtl. visializations)
  - slideshow : everything slide show related
  - pipline? (probably not needed)
  

