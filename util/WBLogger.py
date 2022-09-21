import wandb
import logging

class LogerWB:
    def __init__(self, name, print_messages=False):
        wandb.login(key='3f6ad4849bb16a8098dd3cf40179085a04a731f6')
        wandb.init(project=name, entity="redwing2002")
        self.print_messages = print_messages

    #Train loss adversarial
    def log_loss_epoch_train_adversarial(self, epoch, loss):
        wandb.log({"epoch_train_loss_adversarial": loss})

        if(self.print_messages):
            logging.info('Epoch[' + str(epoch) + '] loss_adversarial:' + str(loss))

    def log_loss_batch_train_adversarial(self, length, epoch, batch_id, loss):
        wandb.log({"batch_train_loss_adversarial": loss})

        if(self.print_messages):
            logging.debug('Epoch[' + str(epoch) + '] ' + 'Finished[' + str(batch_id / length * 100)[:5] + '%]' + 'Batch[' + str(batch_id) + ']' + 'loss_adversarial:' + str(loss))

    #epoch
    def log_epoch(self, epoch):
            wandb.log({"round_": epoch})
            
    #current_iter
    def log_current_iter_epoch(self, iter):
            wandb.log({"current_iter_epoch": iter})
            
    #Train acc adversarial
    def log_acc_epoch_train_adversarial(self, epoch, loss):
            wandb.log({"epoch_train_acc_adversarial": loss})

            if(self.print_messages):
                logging.info('Epoch[' + str(epoch) + '] acc_adversarial:' + str(loss))

    def log_acc_batch_train_adversarial(self, length, epoch, batch_id, loss):
        wandb.log({"batch_train_acc_adversarial": loss})

        if(self.print_messages):
            logging.debug('Epoch[' + str(epoch) + '] ' + 'Finished[' + str(batch_id / length * 100)[:5] + '%]' + 'Batch[' + str(batch_id) + ']' + 'acc_adversarial:' + str(loss))

    #Train iou adversarial
    def log_iou_epoch_train_adversarial(self, epoch, loss):
        wandb.log({"epoch_train_iou_adversarial": loss})

        if(self.print_messages):
            logging.info('Epoch[' + str(epoch) + '] iou_adversarial:' + str(loss))

    def log_iou_batch_train_adversarial(self, length, epoch, batch_id, loss):
        wandb.log({"batch_train_iou_adversarial": loss})

        if(self.print_messages):
            logging.debug('Epoch[' + str(epoch) + '] ' + 'Finished[' + str(batch_id / length * 100)[:5] + '%]' + 'Batch[' + str(batch_id) + ']' + 'iou_adversarial:' + str(loss))

    #Vall loss
    def log_loss_epoch_val(self, epoch, loss):
        wandb.log({"epoch_val_loss": loss})

        if(self.print_messages):
            logging.info('Val[' + str(epoch) + '] loss:' + str(loss))

    #Vall iou
    def log_iou_epoch_val(self, epoch, loss):
        wandb.log({"epoch_val_iou": loss})

        if(self.print_messages):
            logging.info('Val_iou[' + str(epoch) + ' iou:' + str(loss))

    #Vall acc
    def log_acc_epoch_val(self, epoch, loss):
        wandb.log({"epoch_val_acc": loss})

        if(self.print_messages):
            logging.info('Val_iou[' + str(epoch) + ' acc:' + str(loss))

    #Vall loss_adversarial
    def log_loss_epoch_val_adversarial(self, epoch, loss):
        wandb.log({"epoch_val_loss_adversarial": loss})

        if(self.print_messages):
            logging.info('Val[' + str(epoch) + '] loss_adversarial:' + str(loss))

    #Vall iou_adversarial
    def log_iou_epoch_val_adversarial(self, epoch, loss):
        wandb.log({"epoch_val_iou_adversarial": loss})

        if(self.print_messages):
            logging.info('Val_iou[' + str(epoch) + ' iou_adversarial:' + str(loss))

    #Vall acc_adversarial
    def log_acc_epoch_val_adversarial(self, epoch, loss):
        wandb.log({"epoch_val_acc_adversarial": loss})

        if(self.print_messages):
            logging.info('Val_iou[' + str(epoch) + ' acc_adversarial:' + str(loss))

