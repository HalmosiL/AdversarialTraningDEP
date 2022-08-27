from executor.Adversarial import model_immer_attack_auto_loss
import torch

def run(id_, batch, device, model, attack, number_of_steps, data_queue, split, split_size, gen=True):
    print("Gen_", id_, " started..")

    image = batch[0].to(device)

    if(gen):
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
            print("save:", data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
            print("save:", data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
            torch.save(image[i].cpu().detach().clone(), data_queue + 'image_' + str(id_) + '_' + str(i) + '_.pt')
            torch.save(label[i].cpu().detach().clone(), data_queue + 'label_' + str(id_) + '_' + str(i) + '_.pt')
