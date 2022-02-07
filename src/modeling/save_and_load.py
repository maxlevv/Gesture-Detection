from __future__ import annotations
import numpy as np 
from pathlib import Path
# from neural_network import FCNN
from feature_scaling import StandardScaler
import json
from typing import List, Tuple
from metaData import MetaData
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neural_network import FCNN




# def save_meta_json_and_weights(neural_net:FCNN, meta_data:MetaData, save_runs_folder_path:Path, run_group_name:str):
#     run_group_folder_path = save_runs_folder_path / run_group_name
#     if not (run_group_folder_path).is_dir():
#         run_group_folder_path.mkdir()
#     meta_data.run_number = len([item for item in run_group_folder_path.iterdir()])
#     run_folder_path = run_group_folder_path / meta_data.generate_save_run_folder_name()
#     run_folder_path.mkdir()
#     meta_data.to_json(run_folder_path)
#     for i, w in enumerate(neural_net.W):
#         np.save(run_folder_path / f"w_{i}.npy", w)

def save_run(save_runs_folder_path:Path, run_group_name, neural_net:FCNN, author, data_file_name, lr, batch_size, epochs, num_samples, description=None, name=None):
    meta_data = MetaData.from_neural_net(neural_net, author, data_file_name, lr, batch_size, epochs, num_samples, description, name)
    save_meta_json_and_weights(neural_net, meta_data, save_runs_folder_path, run_group_name)

def load_run(from_file_path:Path) -> Tuple[List[np.array], MetaData]:
    # currently this function does nothing but calling another function, some functionality can be added here later
    W, meta_data_dict = load_meta_json_and_weights(from_file_path)
    return W, MetaData.from_dict(meta_data_dict)


def save_meta_json_and_weights(neural_net:FCNN, meta_data:MetaData, save_runs_folder_path:Path, run_group_name:str):
    run_group_folder_path = save_runs_folder_path / run_group_name
    if not (run_group_folder_path).is_dir():
        run_group_folder_path.mkdir()
    meta_data.run_number = len([item for item in run_group_folder_path.iterdir()])
    run_folder_path = run_group_folder_path / meta_data.generate_save_run_folder_name()
    run_folder_path.mkdir()
    meta_data.to_json(run_folder_path)
    for i, w in enumerate(neural_net.W):
        np.save(run_folder_path / f"w_{i}.npy", w)


def load_meta_json_and_weights(from_folder_path:Path) -> Tuple[List[np.array], dict]:
    # loads the weights and meta data from the folder specified and returns them
    W = []
    for w_file_path in from_folder_path.glob(f"w_*.npy"):
        W.append(np.load(w_file_path))
        # here the order of the files could go wrong but practically it should work

    with open(list(from_folder_path.glob('*_meta.json'))[0], 'r') as meta_json_file:
        meta_data_dict = json.load(meta_json_file)
        
    return W, meta_data_dict



