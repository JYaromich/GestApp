from curses import window
from pathlib import Path
from time import sleep
from unittest import result
import numpy as np
import cv2 as cv
import torch
import torchvision
from torchvision import models
from zmq import device
from facenet_pytorch import MTCNN
from Models.GestAppNN import GestRecogNN, GestRecogBlock
from Models import GestAppNN
from matplotlib import pyplot as plt

class DoesnExistCamException(Exception):
    pass

class StreamEndException(Exception):
    pass

class UncorrectPredictionException(Exception):
    pass

class GestApp():
    IMAGE_SIZE = (224, 224)
    FACE_TRESHOLD = 0.7
    THRESHHOLD = 0.3
    
    def __init__(self, path2model: Path) -> None:
        self.path2model = path2model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gest_model = torch.load(self.path2model, map_location=torch.device(self.device))
        self.gest_model.eval()
        self.face_model = MTCNN()
        self.unheandled_image_count = 3
        self.static_image_criterion = 260000.0
        self.past_kl = 0
                                      
    
    def _open_cam(self, cam_number: int = 0):
        cap = cv.VideoCapture(cam_number)
        if not cap.isOpened():
            raise DoesnExistCamException('Connot open camera')
            
        return cap
    
    def _is_person(self, frame):
        _, probs = self.face_model.detect(frame, landmarks=False)
        if probs.any(): 
            return probs.any() > self.FACE_TRESHOLD
        return False
    
    def __change_brithness_contrast_balanse(self, frame, beta, alpha, gamma):
        new_image = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        look_up_table = np.empty((1,256), np.uint8)
        for i in range(256):
            look_up_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv.LUT(new_image, look_up_table)

    def _preparation_frame(self, frame):
        frame = cv.resize(frame, self.IMAGE_SIZE, interpolation = cv.INTER_AREA)
        # frame = self.__change_brithness_contrast_balanse(frame, beta=0, alpha=1, gamma=5)
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, -1).repeat(3, -1)
        frame = frame / 255
        frame = frame.astype('float32')
        return frame
    
    def _predict_gest(self, frame) -> int: 
        prepareated_frame = self._preparation_frame(frame)
        
        tensor_frame = torchvision.transforms.ToTensor()(prepareated_frame)
        output = self.gest_model(tensor_frame[None, :].to(self.device))
        with torch.no_grad(): probabilites = torch.nn.Softmax(dim=-1)(output)

        max_prob = torch.max(probabilites, dim=1)
        
        indx = None
        if max_prob.values > self.THRESHHOLD: 
            indx = torch.max(probabilites, dim=1).indices.item()
        
        return indx
         
    def handling(self, gest_id, frame):
        
        text = str(GestAppNN.INDX2LABEL[gest_id]) if gest_id else None
            
            
        position = (10, 70)
        font_family = cv.FONT_HERSHEY_SIMPLEX
        font_size = 3
        font_colot = (255, 0, 0)
        
        text_params = (text, position, font_family, font_size, font_colot)
        if text:
            frame = cv.putText(frame, *text_params)
            return frame
            
        return np.array([None])    
        
    def _handling_frame(self, frame):
        try:
            gest_id = self._predict_gest(frame)
            handled_frame = self.handling(gest_id, frame)           
            return handled_frame

        except Exception as e:
            print(e)

    def _is_static_image(self, zero_frame, last_frame):
        with torch.no_grad():
            kl = (-1) * torch.nn.functional.kl_div(input=torch.tensor(zero_frame, device=self.device), 
                                            target=torch.tensor(last_frame, device=self.device), 
                                            reduction='batchmean')
            
        result = abs(kl.item() - self.past_kl) < self.static_image_criterion
        self.past_kl = kl.item()
        return result
        
    def run(self):
        cap = None
        try:
            cap = self._open_cam()
            i = 0
            zero_frame, last_frame = np.array([0]), np.array([0])
            while True:               
                ret, frame = cap.read()
                if not ret:
                    raise StreamEndException("Can't receive frame (stream end?). Exiting ...")
                
                if self._is_person(frame=frame):
                    if i == 0: zero_frame = frame.copy()
                    if i == self.unheandled_image_count: last_frame = frame.copy()
                cv.imshow('frame', frame)    
                    
                if (zero_frame.shape[0] > 1) and (last_frame.shape[0] > 1):
                    if self._is_static_image(zero_frame, last_frame):
                        handled_frame = self._handling_frame(last_frame.copy())
                        if handled_frame.shape[0] > 1:
                            plt.imshow(handled_frame)
                            plt.show()
                    
                
                if i < self.unheandled_image_count: i = i + 1
                else:
                    zero_frame, last_frame = np.array([0]), np.array([0])
                    i = 0
                
                if cv.waitKey(1) == ord('q'):
                    break
        
        except DoesnExistCamException as e:
            print(e)
        
        except StreamEndException as e:
            print(e)
        
        except Exception as e:
            print(e)
        
        finally:
            if cap:
                cap.release()
                cv.destroyAllWindows()

            

if __name__ == '__main__':
    # https://www.kaggle.com/code/janeyaromich/gb-torch-object-detection - ноутбук для обучения
    
    app = GestApp(Path('GestApp/Models/inception_98.80'))
    app.IMAGE_SIZE = (299, 299) # for inception block
    
    # app = GestApp(Path('GestApp/Models/final_custom_model'))
    app.run()
    
   
    
    
    
    
