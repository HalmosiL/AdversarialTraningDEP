class Config:
    def __init__(self):
        self.config = None

    def confInit(self, conf):    
        if(conf['MODE'] == "val"):
            self.config = {
                'MODE': 'val',
                'Executor_Finished_Train': "True",
                'Executor_Finished_Val': "False"
            }
        elif(conf['MODE'] == "train"):
            self.config = {
                'MODE': 'train',
                'Executor_Finished_Train': "False",
                'Executor_Finished_Val': "True"
            }

    def setMode(self, mode): 
        if(mode == "val"):
            self.config = {
            'MODE': 'val',
            'Executor_Finished_Train': "True",
            'Executor_Finished_Val': "False"
            }
        elif(mode == "train"):
            self.config = {
            'MODE': 'train',
            'Executor_Finished_Train': "False",
            'Executor_Finished_Val': "True"
            }
        elif(mode == "off"):
            self.config = {
            'MODE': 'off',
            'Executor_Finished_Train': "True",
            'Executor_Finished_Val': "True"
            }

    def alertGenerationFinished(self, mode):
        if(mode == "val"):
            self.config = {
            'MODE': 'val',
            'Executor_Finished_Train': "False",
            'Executor_Finished_Val': "True"
            }
        elif(mode == "train"):
            self.config = {
            'MODE': 'train',
            'Executor_Finished_Train': "True",
            'Executor_Finished_Val': "False"
            }
