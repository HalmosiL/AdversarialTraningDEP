import os
import torch
import time
import xmltodict
import sys

class DatasetAdversarial:    
    def __init__(self, con_conf_path, data_queue_path, slice_, mode_):
        self.con_conf_path = con_conf_path
        self.data_queue_path = data_queue_path
        self.slice_ = slice_
        self.mode_ = mode_

    def __len__(self):
        return sys.maxsize
    
    def readConf(self):        
        while(True):
            file_ = open('../configs/config_com.xml', 'r', encoding='utf-8')
            my_xml = file_.read()

            if(len(my_xml) != 0):
                return xmltodict.parse(my_xml)

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
            con_conf = self.readConf()
                
            if(self.mode_ == "train"):
                if(con_conf['root']['Executor_Finished_Train']['#text'] == "True"):
                    return []
                    
            elif(self.mode_ == "val"):
                if(con_conf['root']['Executor_Finished_Val']['#text'] == "True"):
                    return []

            print(image_path)
            print(label_path)
                
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
                    print("waiting for data sice:" + str(0.01 * count_no_data)[:5] + "(s)...", end="\r")

                time.sleep(0.01)
                
        return [image_, label_, remove_queue]
