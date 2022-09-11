import glob
import time
import torch
import os
import math
import sys

from executor.Gen import run
from executor.Adversarial import Cosine_PDG_Adam
import util.Transforms as transform
from dataset.Dataset import SemData

from models.Model import slice_model, load_model_slice
from util.Comunication import Comunication

class Executor:
    def sort_(self, key):
        key = key.split("_")[-1]
        key = key.split(".")[0]

        return int(key)
    
    def __init__(
        self,
        model_cache,
        queue_size_train,
        queue_size_val, 
        data_queue, 
        data_path, 
        batch_size, 
        number_of_steps,
        device,
        num_workers,
        train_batch_size,
        args_dataset,
        step_size,
        clip_size
    ):
        self.mode = None
        self.comunication = Comunication()
        
        try: 
            print("Create data cache...")
            os.mkdir(data_queue)
            os.mkdir(data_queue[:-1] + "_val")
            print("Data cache created successfuly...")
        except OSError as error: 
            print("Data cache alredy exist...")  

        self.split = -1
        self.split_size = 0
            
        if(train_batch_size < batch_size):
            if(batch_size % train_batch_size != 0):
                print("batch_size", batch_size)
                raise ValueError(
                    "The executor batch size should be divisible by the train batch size...." +
                    "\ntrain_batch_size" + str(train_batch_size) +
                    "\nbatch_size" + str(batch_size))
            else:
                self.split = int(batch_size / train_batch_size)
                self.split_size = int(batch_size / self.split)

        self.model_cache = model_cache
        self.queue_size_train = queue_size_train
        self.queue_size_val = queue_size_val
        self.data_path = data_path
        self.model_name = None
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.number_of_steps = number_of_steps
        self.data_queue = data_queue
        self.step_size = step_size
        self.clip_size = clip_size

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transform = transform.Compose([
            transform.RandScale([args_dataset["scale_min"], args_dataset["scale_max"]]),
            transform.RandRotate([args_dataset["rotate_min"], args_dataset["rotate_max"]], padding=mean, ignore_label=args_dataset["ignore_label"]),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='rand', padding=mean, ignore_label=args_dataset["ignore_label"]),
            transform.ToTensor(),
            transform.Normalize(mean, std)])

        train_data = SemData(
            split='train',
            data_root=data_path,
            data_list=args_dataset["train_list"],
            transform=train_transform
        )

        val_transform = transform.Compose([
            transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='center', padding=mean, ignore_label=args_dataset["ignore_label"]),
            transform.ToTensor(),
            transform.Normalize(mean, std)])

        val_data = SemData(
            split='val',
            data_root=data_path,
            data_list=args_dataset["val_list"],
            transform=val_transform
        )

        self.train_data_set_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True
        )

        self.val_data_set_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=int(self.batch_size/2),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

        self.attack = Cosine_PDG_Adam(
            step_size=self.step_size,
            clip_size=self.clip_size
        )
        
    def updateModel(self, model):
        new_model_name = glob.glob(self.model_cache + "*.pt")
        
        if(not len(new_model_name)):
            if(self.model_name is None):
                print("There is no model to use yet...")
                time.sleep(2)
                return None
        else:
            new_model_name.sort(key=self.sort_)
            new_model_name = new_model_name[-1]              

            if(self.model_name != new_model_name):
                self.model_name = new_model_name

                return load_model_slice(new_model_name, self.device)
            else:
                return model

    def data_queue_is_not_full(self, com_conf_mode):
        if(com_conf_mode == "train"):
            return (len(glob.glob(self.data_queue + "*.pt")) / 2) < self.queue_size_train
        elif(com_conf_mode == "val"):
            return (len(glob.glob(self.data_queue[:-1] + "_val/*.pt")) / 2) < self.queue_size_val
            
    def generateTrainData(self, mode, number_of_steps=None):
        if(number_of_steps is None):
            number_of_steps = self.number_of_steps
        
        if(mode == "train"):
            print("Executor start train....")
            iter_ = iter(self.train_data_set_loader)
        elif(mode == "val"):
            print("Executor start val....")
            iter_ = iter(self.val_data_set_loader)

        element_id = 0
        model = None

        data = self.comunication.readConf()
        
        while(True):
            print("Step....")
            model = self.updateModel(model)
            print("Get config....")
            data = self.comunication.readConf()
            
            print("Generate data....")
            if(data['MODE'] == "off"):
                return
            
            if(self.data_queue_is_not_full(data['MODE'])):
                if(model is not None):
                    try:
                        batch = next(iter_)
                            
                        run(
                            id_=element_id,
                            batch=batch,
                            device=self.device,
                            model=model,
                            attack=self.attack,
                            number_of_steps=number_of_steps,
                            data_queue=self.data_queue if data['MODE'] == "train" else self.data_queue[:-1] + "_val/",
                            split=self.split,
                            split_size=self.split_size,
                            gen=False
                        )                               

                        element_id += 1
                    except StopIteration:
                        if(mode == "train"):
                            self.comunication.alertGenerationFinished("train")
                        elif(mode == "val"):
                            self.comunication.alertGenerationFinished("val")
                            
                        return
                else:
                    print("There is no model to use yet....")
                    time.sleep(2)
            else:
                    print("Data queue is full....")
                    time.sleep(2)
    def start(self):
        e = 1
        n_step = 0
        
        while(True):
                print("GET MAIN CONF....")
                data = self.comunication.readConf()
                print("SET MAIN CONF....")
                print(data)
                time.sleep(2)
                
                if(not data['Executor_Finished_Train'] == "True" and data['MODE'] == "train"):
                    print("START TRAIN GEN...")
                    self.generateTrainData("train", n_step)

                self.model_name = None 
                    
                if(not data['Executor_Finished_Val'] == "True" and data['MODE'] == "val"):
                    print("START VAL GEN...")
                    self.generateTrainData("val")
                
                if(data['MODE'] == "off"):
                    print("Stop Executor...")
                    break
                
                if(e % 50 == 0):
                    n_step += 1
                e += 1
