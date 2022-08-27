import glob
import sys
import torch
import os
import json
import numpy as np

sys.path.insert(0, "../")

from util.Optimizer import poly_learning_rate
from models.Model import get_model
from util.Metrics import intersectionAndUnion
from util.WBLogger import LogerWB

def sort_(key):
    key = key.split("_")[-1]
    key = key.split(".")[0]
    
    return int(key)

def clearDataQueue(CONFIG, mode):
    if(mode == "train"):
        if(os.path.exists(CONFIG['DATA_QUEUE'])):
            for filename in glob.glob(CONFIG['DATA_QUEUE'] + "*.pt"):
                os.unlink(filename)
    elif(mode == "val"):
        if(os.path.exists(CONFIG['DATA_QUEUE'] + "_val/")):
            for filename in glob.glob(CONFIG['DATA_QUEUE'] + "_val/*.pt"):
                os.unlink(filename)


def removeFiles(data):
    remove_files = np.array(data).flatten()
    for m in remove_files:
        os.remove(m)

def setMode(mode):
    with open("../configs/config_com.json", 'r+') as f:
        data_json = json.load(f)
        data_json["MODE"] = mode
        f.seek(0)
        json.dump(data_json, f, indent=4)
        f.truncate() 

def cacheModel(cache_id, model, CONFIG):
    models = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    models.sort(key=sort_)
    torch.save(model.getSliceModel().eval().state_dict(), CONFIG["MODEL_CACHE"] + CONFIG["MODEL_NAME"] + "_" + str(cache_id) + ".pt")

    if len(models) > 5:
        os.remove(models[0])
        
    return cache_id + 1

def train(CONFIG_PATH, CONFIG, train_loader_adversarial, val_loader_adversarial, val_loader):
    logger = LogerWB(CONFIG["WB_LOG"], print_messages=CONFIG["PRINT_LOG"])

    if(CONFIG["MODE_LOADE"]):
        print("Continum Traning.....")
        model = get_model(CONFIG['DEVICE'][0])

        print("Load Model.....")
        model.load_state_dict(torch.load(CONFIG["MODEL_CONTINUM_PATH"]))

        optimizer = torch.optim.SGD(
            [{'params': model.layer0.parameters()},
            {'params': model.layer1.parameters()},
            {'params': model.layer2.parameters()},
            {'params': model.layer3.parameters()},
            {'params': model.layer4.parameters()},
            {'params': model.ppm.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
            {'params': model.cls.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
            {'params': model.aux.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10}],
            lr=CONFIG['LEARNING_RATE'], momentum=CONFIG['MOMENTUM'], weight_decay=CONFIG['WEIGHT_DECAY'])

        print("Load optimizer.....")
        optimizer.load_state_dict(torch.load(CONFIG["OPTIMIZER_CONTINUM_PATH"]))

        print("Traning started.....")
    else:
        model = get_model(CONFIG['DEVICE'][0])
        optimizer = torch.optim.SGD(
            [{'params': model.layer0.parameters()},
            {'params': model.layer1.parameters()},
            {'params': model.layer2.parameters()},
            {'params': model.layer3.parameters()},
            {'params': model.layer4.parameters()},
            {'params': model.ppm.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
            {'params': model.cls.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10},
            {'params': model.aux.parameters(), 'lr': CONFIG['LEARNING_RATE'] * 10}],
            lr=CONFIG['LEARNING_RATE'], momentum=CONFIG['MOMENTUM'], weight_decay=CONFIG['WEIGHT_DECAY'])
        print("Traning started.....")
    
    cache_id = 0
    cache_id = cacheModel(cache_id, model, CONFIG)
    
    max_iter = CONFIG["EPOCHS"] * len(train_loader_adversarial)
    cut_all = 0

    for e in range(CONFIG["EPOCHS"]):
        model = model.train()

        loss_train_epoch = 0
        iou_train_epoch = 0
        acc_train_epoch = 0

        clearDataQueue(CONFIG, "train")
        
        cut_ = 0
        
        print("Train Adversarial loader length:", len(train_loader_adversarial))
        print("Val Adversarial loader length:", len(val_loader_adversarial))
        
        for batch_id, data in enumerate(train_loader_adversarial):
            if(len(data) == 3):
                image = data[0][0].to(DEVICE)
                target = data[1][0].to(DEVICE)
                
                current_iter = e * len(train_loader_adversarial) + batch_id + 1 - cut_all
                poly_learning_rate(optimizer, CONFIG['LEARNING_RATE'], current_iter, max_iter, power=CONFIG['POWER'])
                
                remove_files = np.array(data[2]).flatten()
                optimizer.zero_grad()
                
                output, main_loss, aux_loss, _ = model(image, target)
                loss = main_loss + CONFIG['AUX_WEIGHT'] * aux_loss
                
                loss.backward()
                optimizer.step()

                intersection, union, target = intersectionAndUnion(output, target, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()

                iou = np.mean(intersection / (union + 1e-10))
                acc = sum(intersection) / (sum(target) + 1e-10)

                logger.log_loss_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id + 1, loss.item())
                logger.log_iou_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id + 1, iou)
                logger.log_acc_batch_train_adversarial(train_loader_adversarial.__len__(), e, batch_id + 1, acc)

                iou_train_epoch += iou
                loss_train_epoch += loss.item()
                acc_train_epoch += acc

                if(e % CONFIG["MODEL_CACHE_PERIOD"] == 0):
                    cache_id = cacheModel(cache_id, model, CONFIG)

                removeFiles(remove_files)
            else:
                print("Jump..")
                remove_files = np.array(data[0]).flatten()
                removeFiles(remove_files)
                
                cut_ += 1
                cut_all += 1

        loss_train_epoch = loss_train_epoch / (train_loader_adversarial.__len__() - cut_)
        iou_train_epoch = iou_train_epoch / (train_loader_adversarial.__len__() - cut_)
        acc_train_epoch = acc_train_epoch / (train_loader_adversarial.__len__() - cut_)

        logger.log_loss_epoch_train_adversarial(e, loss_train_epoch)
        logger.log_iou_epoch_train_adversarial(e, iou_train_epoch)
        logger.log_acc_epoch_train_adversarial(e, acc_train_epoch)

        torch.save(model.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_" + str(e) + ".pt")
        torch.save(optimizer.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_optimizer" + str(e) + ".pt")

        cache_id = cacheModel(cache_id, model, CONFIG)
        setMode("val")

        model = model.eval()

        loss_val_epoch = 0
        iou_val_epoch = 0
        acc_val_epoch = 0

        val_status = 0

        clearDataQueue(CONFIG, "val")
        
        print("Val finished:" + str(val_status / val_loader_adversarial.__len__())[:5] + "%", end="\r")
        cut_ = 0
        for data in val_loader_adversarial:
            with torch.no_grad():
                if(len(data) == 3):
                    image_val = data[0][0].to(DEVICE)
                    target = data[1][0].to(DEVICE)
                    remove_files = np.array(data[2]).flatten()

                    output, _ = model(image_val)

                    intersection, union, target = intersectionAndUnion(output, target, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()

                    iou = np.mean(intersection / (union + 1e-10))
                    acc = sum(intersection) / (sum(target) + 1e-10)

                    iou_val_epoch += iou
                    loss_val_epoch += loss
                    acc_val_epoch += acc
                    val_status += 1
                    
                    print("Val finished:" + str(val_status / (val_loader_adversarial.__len__() - cut_))[:5] + "%", end="\r")
                    removeFiles(remove_files)
                else:
                    print("jump...")
                    remove_files = np.array(data[0]).flatten()
                    removeFiles(remove_files)
                    cut_ = cut_ + 1
                
        loss_val_epoch = loss_val_epoch / (val_loader_adversarial.__len__() - cut_)
        iou_val_epoch = iou_val_epoch / (val_loader_adversarial.__len__() - cut_)
        acc_val_epoch = acc_val_epoch / (val_loader_adversarial.__len__() - cut_)

        logger.log_loss_epoch_val_adversarial(e, loss_val_epoch)
        logger.log_iou_epoch_val_adversarial(e, iou_val_epoch)
        logger.log_acc_epoch_val_adversarial(e, acc_val_epoch)

        setMode("train")
