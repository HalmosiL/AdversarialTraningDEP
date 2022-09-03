import os
import torch
import time
import sys

from util.Comunication import Comunication

class DatasetAdversarial:    
    def __init__(self, con_conf_path, data_queue_path, slice_, mode_, host_, port_):
        self.con_conf_path = con_conf_path
        self.data_queue_path = data_queue_path
        self.slice_ = slice_
        self.mode_ = mode_
        self.comunication = Comunication()
        
    def __len__(self):
        return sys.maxsize

    def __getitem__(self, idx):
        image_ = None
        label_ = None
        
        path_a = int(idx / self.slice_)
        path_b = idx % self.slice_
        
        count_no_data = 0
        image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
        label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

        remove_queue = []
        
        while(label_ is None):
            con_conf = self.comunication.readConf()
                
            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                try:
                    count_no_data = 0
                    image_ = torch.load(image_path).clone()
                    label_ = torch.load(label_path).clone()
                    remove_queue.append([image_path, label_path])
                except Exception as e:
                    return [remove_queue]
            else:
                count_no_data += 1
                if(count_no_data > 1 and count_no_data % 200 == 0):
                    print("waiting for data sice:" + str(0.01 * count_no_data)[:5] + "(s)...")
                if(
                    (self.mode_ == "train" and con_conf['Executor_Finished_Train'] == "True") or
                    (self.mode_ == "val" and con_conf['Executor_Finished_Val'] == "True")
                ):
                    return []
                
                time.sleep(0.01)
                
        return [image_, label_, remove_queue]
