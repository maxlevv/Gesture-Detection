from __future__ import annotations
from pathlib import Path
from datetime import date
import json
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neural_network import FCNN


class MetaData():
    def __init__(self, architecture, bias_list, train_acc, scaler, val_acc, loss_function, activation_functions, 
                       author, data_file_name, lr, batch_size, epochs, num_samples, description=None, name=None,
                       run_number=None, date_iso=None):
    
        self.name = name
        self.run_number = run_number # will be determined when saving by counting the number of items in the run_group folder
        self.data_file_name = data_file_name
        if date_iso:
            self.date_iso = date_iso
        else:
            self.date_iso = date.today().isoformat()
        self.author = author
        self.description = description
        self.architecture = architecture
        self.bias_list = bias_list
        self.train_acc = train_acc
        self.scaler = scaler
        self.val_acc = val_acc
        self.loss_function = loss_function
        self.activation_functions = activation_functions
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples

    @classmethod
    def from_neural_net(cls, neural_net:FCNN, author, data_file_name, lr, batch_size, epochs, num_samples, description=None, name=None):
        return cls(
            architecture = [neural_net.input_size] + neural_net.layer_list, 
            bias_list = neural_net.bias_list, 
            train_acc = neural_net.acc_hist[-1],
            scaler = neural_net.scaler.to_dict(), 
            val_acc = neural_net.val_acc, 
            loss_function = neural_net.loss_func_str, 
            activation_functions = neural_net.activation_func_string_list, 
            author = author, 
            data_file_name = data_file_name, 
            lr = lr, 
            batch_size = batch_size, 
            epochs = epochs, 
            num_samples = num_samples, 
            description = description, 
            name = name
        )
    
    @classmethod
    def from_dict(cls, meta_data_dict:dict):
        return cls( **meta_data_dict)


    def to_json(self, folder_path:Path):
        with open(folder_path / (self.generate_save_run_folder_name() + "_meta.json"), 'w') as meta_json_file:
            json.dump(self.__dict__, meta_json_file)
    
    
    def generate_save_run_folder_name(self):
        return str(self.date_iso) + "_" + str(self.run_number) + "_" + "-".join([str(layer) for layer in self.architecture])