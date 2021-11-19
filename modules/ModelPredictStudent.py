import numpy as np
import cv2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from math import sqrt
from modules import width
from modules import height
from modules import channels
from modules import getInformation

class ModelPredictStudent:
  have_height = True
  modelcheckpoint = ModelCheckpoint(
    'checkpoint_best.hdf5',
    save_best_only=True,
    save_weights_only=True
  )
  
  imagedatagenerator = ImageDataGenerator(
    brightness_range=(0, 10),
    rotation_range=(1, 10),
    channel_shift_range=(1, 10),
    horizontal_flip=True
  )
  
  def __init__(self, file=''):
    self.extract_feature = self.__build_layer_extract_feature()
    self.model = self.__build_model()  
    if ModelPredictStudent.have_height:
      print ('Load weight complete')
      self.model.load_weights(file)
    
  def __build_model(self):
    model = self.extract_feature
    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(getInformation.config['students'], activation='softmax'))
    
    model.summary()
    return model 
  
  def __build_layer_extract_feature(self):
    model = Sequential()
    
    model.add(Conv2D(32, (2, 2), padding='same', input_shape=(width, height, channels)))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    
    model.add(Conv2D(8, (2, 2)))
    model.add(Conv2D(8, (2, 2)))
    model.add(Conv2D(8, (2, 2)))
    model.add(Conv2D(8, (2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    
    model.add(Flatten())
    
    return model
  
  def train_model(self, epochs=5):
    X_train, y_train = getInformation.load_data_training()
    
    print ('X_train.shape = {}. y_train.shape = {}'.format(X_train.shape, y_train.shape))
    self.model.compile(
      optimizer='sgd', 
      loss='categorical_crossentropy', 
      metrics=['acc']
    )
    
    print ('Model prepares to train data completely!')
    self.model.fit(
      X_train,
      y_train,
      shuffle=True,
      epochs=epochs,
      callbacks=[self.modelcheckpoint],
      use_multiprocessing=True,
      validation_data=(X_train, y_train)
    )
    print ('Model trains completely!')
  
  def __cosine_similarity(self, face_img, temp_img):
    sum_ = 0
    sum_face_img = 0
    sum_face_temp = 0
    for i, j in zip(face_img, temp_img):
      sum_face_img += i * i
      sum_face_temp += j * j
      sum_ += i * j
    cosine_sum = sum_ / (sqrt(sum_face_img) * sqrt(sum_face_temp))
    return cosine_sum
    
  def predict(self, face_img, len_fls_students=0):
    check_call_for = True
    img = cv2.resize(face_img, (width, height))
    
    result = self.model.predict(np.array([img]))[0]
    index = np.argmax(result)
    counter = 0
    face_img_cosine = self.extract_feature.predict(np.array([img]))[0]
    for name in list(getInformation.symbols.keys()):
      face_img_temp = self.extract_feature.predict(np.array([getInformation.symbols[name]]))[0]
      cosine_similarity = self.__cosine_similarity(list(face_img_temp), list(face_img_cosine))
      if cosine_similarity <= 0.6:
        counter += 1
      
    if counter == len(getInformation.symbols.keys()) or counter == 0:
      return 0

    check_call_for = False if len_fls_students <= 0 else True
    if check_call_for == False:
      return 0
    return index