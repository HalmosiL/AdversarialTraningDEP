import json
import socket
import time

class Comunication:
    tcp_socket = None

    def send(self, data):
        while(True):
            Comunication.tcp_socket.sendall(data.encode())
            if(data != 'GET_CONF'):
                Comunication.tcp_socket.sendall('GET_CONF'.encode())

            response = Comunication.tcp_socket.recv(4096).decode()
            response = json.loads(response)
            if(response != "RESEND"):
                return response

    def readConf(self):
        return self.send('GET_CONF')
    
    def alertGenerationFinished(self, mode):
        while(True):
            if(mode == "train"):
                conf = self.send('ALERT_TRAIN')
                if(conf['MODE'] == 'train' and conf['Executor_Finished_Train'] == "True" and conf['Executor_Finished_Val'] == "False"):
                    return conf
            elif(mode == "val"):
                conf = self.send('ALERT_VAL')
                if(conf['MODE'] == 'val' and conf['Executor_Finished_Train'] == "False" and conf['Executor_Finished_Val'] == "True"):
                    return conf
        
    def setMode(self, mode):
        while(True):
            if(mode == "train"):
                conf = self.send('SET_MODE_TRAIN')
                if(conf['MODE'] == 'train'):
                    return conf
            elif(mode == "val"):
                conf = self.send('SET_MODE_VAL')
                if(conf['MODE'] == 'val'):
                    return conf
