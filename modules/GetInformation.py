from modules import width
from modules import height
import cv2
import numpy as np
import os
import json

class GetInformation:
  path = 'data.json'
  dataset = 'dataset'
  number_of_students = 0
  
  students = dict()
  
  def __init__(self):
    with open(GetInformation.path, "rb") as fin:
      data = json.load(fin, encoding='utf-8')
    self.data = data['data']
    self.config = data['config']
    self.symbols = {}
      
    GetInformation.number_of_students = len(self.data.keys())
    # print ('Number of students = {}'.format(GetInformation.number_of_students))
    # print (self.data.get('Nguyen Duc_Bao Lam'))
    self.__load_data_symbols()
    
  def load_data_training(self):
    X = []
    y = []
    folder_names = os.listdir(GetInformation.dataset)
    
    for folder in folder_names:
      print ('Load student name = {}'.format(folder))
      for file in os.listdir(os.path.join(GetInformation.dataset, folder)):
        img = cv2.imread(os.path.join(GetInformation.dataset, folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (width, height))
        y_ = np.zeros((self.config['students'],))
        y_[self.data[folder]['id']] = 1
        y.append(y_)
        X.append(img)
        print ('{} hanles successfully!'.format(os.path.join(GetInformation.dataset, folder, file)))
        
    return np.array(X), np.array(y)
  
  def get_information(self, id):
    for idx, name in enumerate(self.data):
      if self.data[name]['id'] == id:
        return name, self.data[name]['class']
    return '', ''
  
  def build_information(self, name, class_):
    if self.data.get(name) == None:
      GetInformation.number_of_students += 1
    
      self.data[name] = {
        'class' : class_,
        'id' : GetInformation.number_of_students 
      }
      
      with open(GetInformation.path, 'w', encoding='utf-8') as fout:
        json.dump({
          'data' : self.data,
          'config' : self.config
        }, fout, ensure_ascii=False)
      
      self.__load_data_symbols()
        
  def __load_data_symbols(self):
    folder_names = os.listdir(GetInformation.dataset)
    
    for folder in folder_names:
      file = os.listdir(os.path.join(GetInformation.dataset, folder))[0]
      img = cv2.imread(os.path.join(GetInformation.dataset, folder, file))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, (width, height))
      self.symbols[folder] = img