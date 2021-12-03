import threading
import cv2
import os
import time

from modules import getInformation
from modules import model_predict_student

class DetectPersons:
  video = cv2.VideoCapture(0)
  detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  total_file_collect = 250
    
  def read_camera(self):
    while True:
      __, img = DetectPersons.video.read()
      
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = DetectPersons.detector.detectMultiScale(gray, 1.3, 5)
      
      if len(faces) == 0:
        pass
      else:
        for (x, y, w, h) in faces:
          face_img = gray[y:y+h, x:x+w]      
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          
          result = model_predict_student.predict(face_img, len(os.listdir(getInformation.dataset)))
          
          if result == 0:
            cv2.destroyAllWindows()
            name = input("MỜI BẠN NHẬP TÊN VÔ:")
            class_ = input("MỜI BẠN NHẬP TÊN LỚP:")
            getInformation.build_information(name, class_)
            self.data_collecting(name)
            print ('THU THẬP DỮ LIỆU HOÀN TẤT')
            print ('BẮT ĐẦU VÔ QUÁ TRÌNH TRAIN PHÂN LOẠI')
            model_predict_student.train_model()
          else:
            name, __ = getInformation.get_information(result)
            cv2.putText(img, name, 
              (20, 20), 
              cv2.FONT_HERSHEY_SIMPLEX,
              1,
              (0,0,255),
              2,
              cv2.LINE_AA
            )
                                                       
      cv2.imshow("TESTING", img)    
      if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    DetectPersons.video.release()
    cv2.destroyAllWindows()
    
  def data_collecting(self, name):
    continous = 0
    checking = False
    while continous < DetectPersons.total_file_collect:
      __, img = DetectPersons.video.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = DetectPersons.detector.detectMultiScale(gray, 1.3, 5)
      if len(faces) == 0:
        cv2.putText(img, 'KHÔNG PHÁT HIỆN THẤY KHUÔN MẶT', 
          (20, 20), 
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0,0,255),
          2,
          cv2.LINE_AA
        )
      elif len(faces) > 1:
        assert checking == True
        cv2.putText(img, 'CÓ NHIỀU HƠN 2 NGƯỜI',
          (20, 20), 
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0,0,255),
          2,
          cv2.LINE_AA
        )
      else:
        x, y, w, h = faces[0]
        face_gray = gray[y : y + h, x : x + w]
        gray = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
        folder = os.path.join(getInformation.dataset, name)
        try:
          os.makedirs(folder)
        except:
          pass
        file = '{}\{}.png'.format(folder, time.time())
        cv2.imwrite(file, face_gray)
        continous += 1
        print ('File = {}. Counter = {}'.format(file, continous))
      cv2.imshow("DETECT_FACE", gray)
      if cv2.waitKey(100) & 0xFF == ord('q'):
        break
      checking = True
    # DetectPersons.video.release()
    cv2.destroyAllWindows()