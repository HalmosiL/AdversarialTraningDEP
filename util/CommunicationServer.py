import socket
import sys
import json
import threading
import sys

sys.path.append('./')

import Config

config = Config.Config()

def readConfig(path):
    with open (path, "r") as f:
        return json.loads(f.read())

def serverInit(conf):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    server_address = ('localhost', conf["CONFMANAGER_PORT"])
    server.bind(server_address)

    server.listen(2)
    return server

def handle_client(client_socket):
    while(True):
        request = client_socket.recv(1024).decode()

        switcher = {
            "GET_CONF": lambda : client_socket.send(json.dumps(config.config, indent = 4).encode()),
            "SET_MODE_VAL": lambda : config.setMode("val"),
            "SET_MODE_TRAIN": lambda : config.setMode("train"),
            "SET_MODE_OFF": lambda : config.setMode("off"),
            "ALERT_TRAIN": lambda : config.alertGenerationFinished("train"),
            "ALERT_VAL": lambda : config.alertGenerationFinished("val")
            }

        value = switcher.get(request, "default")
        if(value == "default"):
            client_socket.send(json.dumps("RESEND", indent = 4).encode())
        else:
            value()

def start(server):
    while True:
        client, addr = server.accept()
        print("[*] Accepted connection from: %s:%d" + str(addr[0]), str(addr[1]))
        client_handler = threading.Thread(target = handle_client, args=(client,))
        client_handler.start()

if __name__ == "__main__":
    conf = readConfig(sys.argv[1])
    print("Read Config...")
    config.confInit(conf)
    print("Init Server...")
    server = serverInit(conf)
    print("Start Server...")
    start(server)
