import os

class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance
  
class Comunication(SingletonClass):
  def readConf(self):    
    return {
        'MODE': os.environ['MODE'],
        'Executor_Finished_Train': os.environ['Executor_Finished_Train'],
        'Executor_Finished_Val': os.environ['Executor_Finished_Val']
    }
    
  def alertGenerationFinished(self, mode):
    if(mode == "train"):
        os.environ['MODE'] = 'train'
        os.environ['Executor_Finished_Train'] = "True"
        os.environ['Executor_Finished_Val'] = "False"
    elif(mode == "val"):
        os.environ['MODE'] = 'val'
        os.environ['Executor_Finished_Train'] = "False"
        os.environ['Executor_Finished_Val'] = "True"
          
  def conConfInit(self, mode):
    if(mode == "val"):
        os.environ['MODE'] = 'val'
        os.environ['Executor_Finished_Train'] = "True"
        os.environ['Executor_Finished_Val'] = "False"
    elif(mode == "train"):
        os.environ['MODE'] = 'train'
        os.environ['Executor_Finished_Train'] = "False"
        os.environ['Executor_Finished_Val'] = "True"
        
  def setMode(self, mode): 
    if(mode == "val"):
        os.environ['MODE'] = 'val'
        os.environ['Executor_Finished_Train'] = "True"
        os.environ['Executor_Finished_Val'] = "False"
    elif(mode == "train"):
        os.environ['MODE'] = 'train'
        os.environ['Executor_Finished_Train'] = "False"
        os.environ['Executor_Finished_Val'] = "True"
                
