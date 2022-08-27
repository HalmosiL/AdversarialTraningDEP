from torch.autograd import Variable
import torch

class Adam_optimizer:
    def __init__(self, B1, B2, lr):
        self.B1 = B1
        self.B2 = B2
        self.lr = lr

        self.m_t = 0
        self.v_t = 0

        self.t = 1
        self.e = 1e-08

    def step_grad(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        return (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

    def step(self, grad, image):
        self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
        self.v_t = self.B2 * self.v_t + (1 - self.B2) * (grad ** 2)

        m_l = self.m_t / (1 - self.B1 ** self.t)
        v_l = self.v_t / (1 - self.B2 ** self.t)

        self.t += 1

        image = image - (self.lr * m_l) / (torch.sqrt(self.v_t) + self.e)

        return image

class Cosine_PDG_Adam:
    def __init__(self, step_size, clip_size):
        self.step_size = step_size
        self.clip_size = clip_size
        self.step_size = step_size

        self.optimizer = Adam_optimizer(lr=step_size, B1=0.9, B2=0.99)
        self.loss_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.step_ = 0

    def step(self, image_o, image, prediction, target):
        prediction = prediction.reshape(prediction.shape[0], -1)
        target = target.reshape(prediction.shape[0], -1)

        loss = (1 - self.loss_function(prediction, target + 0.0001)).sum()
        grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
        
        image = self.optimizer.step(-1 * grad, image)
            
        image = torch.max(torch.min(image, image_o + self.clip_size), image_o - self.clip_size)
        image = image.clamp(0,1)

        return image
    
    def reset(self):
        self.optimizer = Adam_optimizer(lr=self.step_size, B1=0.9, B2=0.99)
        

def model_immer_attack_auto_loss(image, model, attack, number_of_steps, device):
    model.zero_grad()
    image_adv = image.clone().detach().to(device)
    image_adv.requires_grad = True
    target = model(image)

    for i in range(number_of_steps):
        prediction = model(image_adv)
        image_adv = attack.step(image, image_adv, prediction, target)
        model.zero_grad()
    
    attack.reset()

    return image_adv