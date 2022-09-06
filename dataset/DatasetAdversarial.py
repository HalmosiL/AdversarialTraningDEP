import os
import torch
import time
import sys

class DatasetAdversarial:    
    def __init__(self, con_conf_path, data_queue_path, slice_, mode_):
        self.con_conf_path = con_conf_path
        self.data_queue_path = data_queue_path
        self.slice_ = slice_
        self.mode_ = mode_
        
    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        image_ = None
        label_ = None
        
        path_a = int(idx / self.slice_)
        path_b = idx % self.slice_

        image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
        label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

        if(
            os.path.exists(image_path) and
            os.path.exists(label_path)
        ):
            try:
                count_no_data = 0
                image_ = torch.load(image_path).clone()
                label_ = torch.load(label_path).clone()
                return [image_.reshape(1, *image_.shape), label_.reshape(1, *label_.shape), [image_path, label_path]]
            except Exception as e:
                print("wrong")
                return [[image_path, label_path]]
                
        return []
