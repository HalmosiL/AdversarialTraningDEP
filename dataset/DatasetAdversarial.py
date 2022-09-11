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
        path_a = int(idx / self.slice_)
        path_b = idx % self.slice_

        if(self.mode_ == "train"):
            image_normal_path = self.data_queue_path + "image_normal" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_normal_path = self.data_queue_path + "label_normal" + str(path_a) + "_" + str(path_b) + "_.pt"
            image_adversarial_path = self.data_queue_path + "image_adversarial" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_adversarial_path = self.data_queue_path + "label_adversarial" + str(path_a) + "_" + str(path_b) + "_.pt"

            if(
                os.path.exists(image_normal_path) and
                os.path.exists(label_normal_path) and
                os.path.exists(image_adversarial_path) and
                os.path.exists(label_adversarial_path)
            ):
                try:
                    image_normal = torch.load(image_normal_path).clone()
                    label_normal = torch.load(label_normal_path).clone()
                    image_adversaria = torch.load(image_adversarial_path).clone()
                    label_adversaria = torch.load(label_adversarial_path).clone()
                    return [
                        image_normal.reshape(1, *image_normal.shape),
                        label_normal.reshape(1, *label_normal.shape),
                        image_adversaria.reshape(1, *image_normal.shape),
                        label_adversaria.reshape(1, *label_normal.shape),
                        [image_normal_path, label_normal_path, image_adversarial_path, label_adversarial_path]
                    ]
                except Exception as e:
                    print("wrong")
                    return [[image_normal_path, label_normal_path, image_adversarial_path, label_adversarial_path]]

            return []
        elif(self.mode_ == "val" or self.mode_ == "off"):
            image_path = self.data_queue_path + "image_" + str(path_a) + "_" + str(path_b) + "_.pt"
            label_path = self.data_queue_path + "label_" + str(path_a) + "_" + str(path_b) + "_.pt"

            if(
                os.path.exists(image_path) and
                os.path.exists(label_path)
            ):
                try:
                    image_ = torch.load(image_path).clone()
                    label_ = torch.load(label_path).clone()
                    return [
                        image_.reshape(1, *image_.shape),
                        label_.reshape(1, *label_.shape),
                        [image_path, label_path]
                    ]
                except Exception as e:
                    print(e)
                    print("wrong")
                    return [[image_path, label_path]]

            return []
