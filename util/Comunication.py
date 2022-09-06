import json
import socket
import time

class Comunication:
    tcp_socket = None

    def send(self, data):
        i = 0
        
        while(True):
            Comunication.tcp_socket.sendall(data.encode())
            if(data != 'GET_CONF'):
                Comunication.tcp_socket.sendall('GET_CONF'.encode())

            response = Comunication.tcp_socket.recv(4096).decode()
            if(response != '"RESEND"'):
                try:
                    return json.loads(response.split("}")[0] + "}")
                except:
                    raise ValueError(response.split("}")[0] + "}")
            else:
                i += 1
                if(i == 6):
                    raise ValueError("Too much wating...")
                time.sleep(2)

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
            elif(mode == "off"):
                conf = self.send('SET_MODE_OFF')
                if(conf['MODE'] == 'off'):
                    return conf
