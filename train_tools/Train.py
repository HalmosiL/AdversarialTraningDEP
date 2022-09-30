import glob
import sys
import torch
import os
import time
import numpy as np
import logging

sys.path.insert(0, "../")

from util.Optimizer import poly_learning_rate
from models.Model import get_model
from util.Metrics import intersectionAndUnion
from util.WBLogger import LogerWB
from util.Comunication import Comunication
from util.AverageMeter import AverageMeter

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
        if(os.path.exists(CONFIG['DATA_QUEUE'][:-1] + "_val/")):
            for filename in glob.glob(CONFIG['DATA_QUEUE'][:-1] + "_val/*.pt"):
                os.unlink(filename)


def removeFiles(data):
    remove_files = np.array(data).flatten()
    for m in remove_files:
        os.remove(m)

def cacheModel(cache_id, model, CONFIG):
    models = glob.glob(CONFIG["MODEL_CACHE"] + "*.pt")
    models.sort(key=sort_)
    torch.save(model.state_dict(), CONFIG["MODEL_CACHE"] + CONFIG["MODEL_NAME"] + "_" + str(cache_id) + ".pt")

    if len(models) > 5:
        os.remove(models[0])
        
    return cache_id + 1

def train(CONFIG_PATH, CONFIG, train_loader_adversarial_, val_loader_adversarial_, val_loader_, start):
    logger = LogerWB(CONFIG["WB_LOG"], print_messages=CONFIG["PRINT_LOG"])
    comunication = Comunication()
    
    if(CONFIG["MODE_LOADE"] == "continum"):
        logging.info("Continum Traning.....")
        model = get_model(CONFIG['DEVICE_TRAIN'])

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

        logging.info("Load optimizer.....")
        optimizer.load_state_dict(torch.load(CONFIG["OPTIMIZER_CONTINUM_PATH"]))

        logging.info("Traning started.....")
    elif(CONFIG["MODE_LOADE"] == "transfer"):
        logging.info("Continum Traning.....")
        model = get_model(CONFIG['DEVICE_TRAIN'])

        print("Load Model.....")
        model.load_state_dict(torch.load(CONFIG["MODEL_CONTINUM_PATH"])["state_dict"])
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

        logging.info("Traning started.....")
    else:
        model = get_model(CONFIG['DEVICE_TRAIN'])
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
        logging.info("Traning started.....")
    
    cache_id = 0
    cache_id = cacheModel(cache_id, model, CONFIG)
    
    max_iter = int(CONFIG["EPOCHS"] * CONFIG["TRAIN_DATASET_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])

    train_loader_len = int(CONFIG["TRAIN_DATASET_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])
    val_loader_len = int(CONFIG["VAL_DATASET_SIZE"] / CONFIG["TRAIN_BATCH_SIZE"])
    
    current_iter = 0
    
    for e in range(CONFIG["EPOCHS"]):
        model = model.train()

        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        clearDataQueue(CONFIG, "train")

        logging.info("Train Adversarial loader length:" + str(train_loader_len))
        logging.info("Val Adversarial loader length:" + str(val_loader_len))
        
        batch_id = 0
        
        data = train_loader_adversarial_.__getitem__(0)
        check_ = 0
        
        while(comunication.readConf()['Executor_Finished_Train'] != "True" or len(data) != 0):
            if(len(data) == 3):
                image_normal = data[0][0].to(CONFIG['DEVICE_TRAIN'])
                target_normal = data[1][0].to(CONFIG['DEVICE_TRAIN'])
                
                logging.debug(image_normal.shape)
                logging.debug(target_normal.shape)

                poly_learning_rate(optimizer, CONFIG['LEARNING_RATE'], current_iter, max_iter, power=CONFIG['POWER'])
                
                remove_files = np.array(data[2]).flatten()
                optimizer.zero_grad()

                output_normal, main_loss, aux_loss, _ = model(image_normal, target_normal)
                loss = main_loss + CONFIG['AUX_WEIGHT'] * aux_loss
                
                loss.backward()
                optimizer.step()

                intersection_normal, union_normal, target_normal = intersectionAndUnion(output_normal, target_normal, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                intersection_normal, union_normal, target_normal = intersection_normal.cpu().numpy(), union_normal.cpu().numpy(), target_normal.cpu().numpy()
                
                intersection_meter.update(intersection_normal), union_meter.update(union_normal), target_meter.update(target_normal)
                loss_meter.update(loss.item(), image_normal.size(0))
                
                iou = np.mean(intersection_normal / (union_normal + 1e-10))
                acc = sum(intersection_normal) / sum(target_normal)

                logger.log_loss_batch_train_adversarial(train_loader_len, e, batch_id + 1, loss.item())
                logger.log_iou_batch_train_adversarial(train_loader_len, e, batch_id + 1, iou)
                logger.log_acc_batch_train_adversarial(train_loader_len, e, batch_id + 1, acc)

                if(e % CONFIG["MODEL_CACHE_PERIOD"] == 0):
                    cache_id = cacheModel(cache_id, model, CONFIG)

                removeFiles(remove_files)
                batch_id += 1
                current_iter += 1
                check_ = 0
                
                logger.log_current_iter_epoch(current_iter)
                logger.log_epoch(int(e))
            elif(len(data) == 1):
                logging.info("Jump..")
                remove_files = np.array(data[0]).flatten()
                removeFiles(remove_files)

                batch_id += 1
                check_ = 0
            else:
                logging.debug("Wait...")
                check_ += 1
                time.sleep(0.5)
                
                if(check_ % 20 == 0):
                    comunication.setMode("train")
                    logging.debug(comunication.readConf())
                
                if(check_ == 60):
                    logging.info("Leave batch...\n")
                    batch_id += 1
                    check_ = 0
                
            data = train_loader_adversarial_.__getitem__(batch_id)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.log_loss_epoch_train_adversarial(e, loss_meter.sum/(batch_id * CONFIG['TRAIN_BATCH_SIZE']))
        logger.log_iou_epoch_train_adversarial(e, mIoU)
        logger.log_acc_epoch_train_adversarial(e, allAcc)

        torch.save(model.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_" + str(e) + ".pt")
        torch.save(optimizer.state_dict(), CONFIG["MODEL_SAVE"] + CONFIG["MODEL_NAME"] + "_optimizer" + str(e) + ".pt")

        cache_id = cacheModel(cache_id, model, CONFIG)

        model = model.eval()

        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        clearDataQueue(CONFIG, "val")
        
        logging.info("Set val...")
        comunication.setMode("val")
        
        data = val_loader_adversarial_.__getitem__(0)
        batch_id = 0
        check_ = 0

        while(comunication.readConf()['Executor_Finished_Val'] != "True" or len(data) != 0):
            with torch.no_grad():
                if(len(data) == 3):
                    image_val = data[0][0].to(CONFIG['DEVICE_TRAIN'])
                    target = data[1][0].to(CONFIG['DEVICE_TRAIN'])
                    remove_files = np.array(data[2]).flatten()

                    output, _, loss = model(image_val, target)

                    intersection, union, target = intersectionAndUnion(output, target, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                    intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                    
                    intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                    loss_meter.update(loss.item(), image_val.size(0))

                    logging.debug("Val finished:" + str(batch_id / (val_loader_len))[:5] + "%")
                    removeFiles(remove_files)
                    batch_id += 1
                elif(len(data) == 1):
                    logging.info("Jump..")
                    remove_files = np.array(data[0]).flatten()
                    removeFiles(remove_files)

                    batch_id += 1
                    check_ = 0
                else:
                    logging.debug("Wait...")
                    check_ += 1
                    time.sleep(0.5)

                    if(check_ % 20 == 0):
                        comunication.setMode("val")
                        logging.info("Leave batch...\n")
                        logging.debug(comunication.readConf())
                    
                    if(check_ == 60):
                        batch_id += 1
                        check_ = 0
                
                data = val_loader_adversarial_.__getitem__(batch_id)
                    
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.log_loss_epoch_val_adversarial(e, loss_meter.sum/(batch_id * CONFIG['TRAIN_BATCH_SIZE']))
        logger.log_iou_epoch_val_adversarial(e, mIoU)
        logger.log_acc_epoch_val_adversarial(e, allAcc)
        
        loss_meter = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        
        batch_id = 0
        
        for data in val_loader_:
            with torch.no_grad():
                image_val, target = data
                image_val = image_val.to(CONFIG['DEVICE_TRAIN'])
                target = target.to(CONFIG['DEVICE_TRAIN'])
                
                output, _, loss = model(image_val, target)

                intersection, union, target = intersectionAndUnion(output, target, CONFIG['CALSSES'], CONFIG['IGNOR_LABEL'])
                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
                loss_meter.update(loss.item(), image_val.size(0))
                batch_id += 1

            logging.debug("Val Normal Finished:" + str(batch_id * 100 / val_loader_.__len__()))
                
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        logger.log_loss_epoch_val(e, loss_meter.sum/(val_loader_.__len__() * 16))
        logger.log_iou_epoch_val(e, mIoU)
        logger.log_acc_epoch_val(e, allAcc)

        comunication.setMode("train")
