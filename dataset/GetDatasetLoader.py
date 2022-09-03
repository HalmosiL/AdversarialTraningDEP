import torchvision.transforms as T
import glob
import json
import sys
import math
import torch
import time

from dataset.DatasetAdversarial import DatasetAdversarial

def getDatasetLoader(CONFIG_PATH, type_="train", num_workers=0, pin_memory=False):
    CONFIG = json.load(open(CONFIG_PATH))

    if(CONFIG["BATCH_SIZE"] % CONFIG["TRAIN_BATCH_SIZE"] != 0):
        raise ValueError('The executor batch size should be divisible by the train batch size....')
    
    slice_ = int(CONFIG["BATCH_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])
    dataset = None    

    if(type_ == "train"):
        dataset = DatasetAdversarial(
            con_conf_path="../configs/config_com.json",
            data_queue_path=CONFIG['DATA_QUEUE'],
            slice_=slice_,
            mode_="train",
            host_=CONFIG["CONFMANAGER_HOST"],
            port_=CONFIG["CONFMANAGER_PORT"]
        )
    else:
        dataset = DatasetAdversarial(
            con_conf_path="../configs/config_com.json",
            data_queue_path=CONFIG['DATA_QUEUE'][:-1]+ "_val/",
            slice_=slice_,
            mode_="val",
            host_=CONFIG["CONFMANAGER_HOST"],
            port_=CONFIG["CONFMANAGER_PORT"]
        )

    return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=pin_memory)
