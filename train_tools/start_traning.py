import subprocess
import json
import sys
import glob
import os
import torch

sys.path.insert(0, "../")
from dataset.GetDatasetLoader import getDatasetLoader
from dataset.Dataset import SemData
from train_tools.Train import train
import util.Transforms as transform

def start(CONFIG_PATH, script):
    bashCommand = [script, CONFIG_PATH]
    list_files = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)

def conConfInit(mode):
    with open("../configs/config_com.json", 'r+') as f:
        data_json = json.load(f)

        if(mode == "train"):
            data_json["MODE"] = "train"
            data_json["Executor_Finished_Train"] = False
            data_json["Executor_Finished_Val"] = True
        elif(mode == "val"):
            data_json["MODE"] = "val"
            data_json["Executor_Finished_Train"] = True
            data_json["Executor_Finished_Val"] = False

        f.seek(0)
        json.dump(data_json, f, indent=4)
        f.truncate() 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You have to give a config file path...')

    CONFIG_PATH = sys.argv[1]
    CONFIG = json.load(open(CONFIG_PATH, "r+"))
    
    print("init com conf...")
    conConfInit(CONFIG["MODE"])

    print("Clear model cache...")
    models_in_cache = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    for m in models_in_cache:
        os.remove(m)

    if(CONFIG["START_EXECUTOR"]):
        start(CONFIG_PATH, "./start_executor.sh")

    train_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="train",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )
    val_loader_adversarial = getDatasetLoader(
        CONFIG_PATH,
        type_="val",
        num_workers=CONFIG["NUMBER_OF_WORKERS_DATALOADER"],
        pin_memory=CONFIG["PIN_MEMORY_ALLOWED_DATALOADER"]
    )

    args_dataset = CONFIG['DATASET']

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transform = transform.Compose([
        transform.Crop([args_dataset["train_h"], args_dataset["train_w"]], crop_type='center', padding=mean, ignore_label=args_dataset["ignore_label"]),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_loader = torch.utils.data.DataLoader(
        dataset=SemData(
            split='val',
            data_root=CONFIG['DATA_PATH'],
            data_list=CONFIG['DATASET']['val_list'],
            transform=val_transform
        ),
        batch_size=CONFIG['TRAIN_BATCH_SIZE'],
        num_workers=CONFIG['NUMBER_OF_WORKERS_DATALOADER'],
        pin_memory=CONFIG['PIN_MEMORY_ALLOWED_DATALOADER']
    )

    train(CONFIG_PATH, CONFIG, train_loader_adversarial, val_loader_adversarial, val_loader)
