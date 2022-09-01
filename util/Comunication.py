from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import xmltodict
  
class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance
  
class Comunication(SingletonClass):
  def readConf(self):   
    while(True):
      try:
        with open('../configs/config_com.xml', 'r', encoding='utf-8') as file:
            my_xml = file.read()

        my_dict = xmltodict.parse(my_xml)
        break
      except Exeption as e:
        print(e)
        
    return {
      'MODE': my_dict['root']['MODE']['#text'],
      'Executor_Finished_Train': my_dict['root']['Executor_Finished_Train']['#text'],
      'Executor_Finished_Val': my_dict['root']['Executor_Finished_Val']['#text']
    }
    
  def alertGenerationFinished(self, mode):
    if(mode == "val"):
        data = {
          'MODE': 'val',
          'Executor_Finished_Train': "False",
          'Executor_Finished_Val': "True"
        }
    elif(mode == "train"):
        data = {
          'MODE': 'train',
          'Executor_Finished_Train': "True",
          'Executor_Finished_Val': "False"
        }
        
    xml = dicttoxml(data)
    xml_decode = xml.decode()

    xmlfile = open("../configs/config_com.xml", "w+")
    xmlfile.write(xml_decode)
    xmlfile.close()
          
  def conConfInit(self, mode):    
    if(mode == "val"):
        data = {
          'MODE': 'val',
          'Executor_Finished_Train': "True",
          'Executor_Finished_Val': "False"
        }
    elif(mode == "train"):
        data = {
          'MODE': 'train',
          'Executor_Finished_Train': "False",
          'Executor_Finished_Val': "True"
        }
        
    xml = dicttoxml(data)
    xml_decode = xml.decode()

    xmlfile = open("../configs/config_com.xml", "w+")
    xmlfile.write(xml_decode)
    xmlfile.close()
        
  def setMode(self, mode): 
    if(mode == "val"):
        data = {
          'MODE': 'val',
          'Executor_Finished_Train': "True",
          'Executor_Finished_Val': "False"
        }
    elif(mode == "train"):
        data = {
          'MODE': 'train',
          'Executor_Finished_Train': "False",
          'Executor_Finished_Val': "True"
        }
        
    xml = dicttoxml(data)
    xml_decode = xml.decode()

    xmlfile = open("../configs/config_com.xml", "w")
    xmlfile.write(xml_decode)
    xmlfile.close()
                
