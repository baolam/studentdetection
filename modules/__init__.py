width = 64
height = 64
channels = 1
from modules.GetInformation import GetInformation

getInformation = GetInformation()

from modules.ModelPredictStudent import ModelPredictStudent

model_predict_student = ModelPredictStudent('checkpoint_best.hdf5')

from modules.DetectPersons import DetectPersons

detectpersons = DetectPersons()
