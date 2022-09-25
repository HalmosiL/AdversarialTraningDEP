from executor.Adversarial import model_immer_attack_auto_loss
from torchvision import transforms
import torch
import logging

def run(id_, batch, device, model, attack, number_of_steps, data_queue, split, split_size, gen=True):
    logging.debug("Gen_" + str(id_) + " started..")
    if(gen):
        image = batch[0].to(device)
        image = torch.split(image, int(len(image)/4))
        image_normal = torch.cat((image[0], image[1], image[2]), dim=0)
        image_adversarial = image[3]
        
        image_adversarial = model_immer_attack_auto_loss(
            image=image_adversarial,
            model=model,
            attack=attack,
            number_of_steps=number_of_steps,
            device=device
        )
        
        label = batch[1]
        label = torch.split(label, int(len(label)/4))
        label_normal = torch.cat((label[0], label[1], label[2]), dim=0)
        label_adversarial = label[3]

        if(split == -1 or split == 1):
            torch.save(torch.cat(image_normal.cpu().detach(), image_adversarial.cpu().detach()), data_queue + 'image_' + str(id_) + '_0_.pt')
            torch.save(torch.cat(label_normal.cpu().detach(), label_adversarial.cpu().detach()), data_queue + 'label_' + str(id_) + '_0_.pt')
        else:
            image_normal = torch.split(image_normal, int(len(image_normal) / 2))
            image_adversarial = torch.split(image_adversarial, int(len(image_adversarial) / 2))
            label_normal = torch.split(label_normal, int(len(image_normal) / 2))
            label_adversarial = torch.split(label_adversarial, int(len(image_adversarial) / 2))

            for i in range(len(image_normal)):
                logging.debug("save:" + data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
                logging.debug("save:" + data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
                torch.save(torch.cat(((image_normal[i].cpu().detach(), image_adversarial[i].cpu().detach()))), data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
                torch.save(torch.cat(((label_normal[i].cpu().detach(), label_adversarial[i].cpu().detach()))), data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
    else:
        image = batch[0].to(device)
        image = model_immer_attack_auto_loss(
            image=image,
            model=model,
            attack=attack,
            number_of_steps=number_of_steps,
            device=device
        )
        
        label = batch[1]
    
        if(split == -1 or split == 1):
            torch.save(image.cpu().detach(), data_queue + 'image_' + str(id_) + '_0_.pt')
            torch.save(label.cpu().detach(), data_queue + 'label_' + str(id_) + '_0_.pt')
        else:
            image = torch.split(image, split_size)
            label = torch.split(label, split_size)
            
            for i in range(len(image)):
                logging.debug("save:" + data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
                logging.debug("save:" + data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
                torch.save(image[i].cpu().detach().clone(), data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
                torch.save(label[i].cpu().detach().clone(), data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
