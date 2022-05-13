from pathlib import Path
import numpy as np
import cv2 as cv
import torch
import torchvision
from torchvision import models
from facenet_pytorch import MTCNN
from Models.GestAppNN import GestRecogNN, GestRecogBlock
from Models import GestAppNN

class DoesnExistCamException(Exception):
    pass

class StreamEndException(Exception):
    pass

class UncorrectPredictionException(Exception):
    pass

class GestApp():
    IMAGE_SIZE = (224, 224)
    FACE_TRESHOLD = 0.7
    
    def __init__(self, path2model: Path) -> None:
        self.path2model = path2model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gest_model = torch.load(self.path2model, map_location=torch.device(self.device))
        self.gest_model.eval()
        self.face_model = MTCNN()
    
    def _open_cam(self, cam_number: int = 0):
        cap = cv.VideoCapture(cam_number)
        if not cap.isOpened():
            raise DoesnExistCamException('Connot open camera')
            
        return cap
    
    def _is_person(self, frame):
        _, probs = self.face_model.detect(frame, landmarks=False)
        if probs > self.FACE_TRESHOLD:
            return True
        return False
    
    def __change_brithness_contrast_balanse(self, frame, beta, alpha, gamma):
        new_image = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv.LUT(new_image, lookUpTable)

    """
    import matplotlib.pyplot as plt
    alpha=1
    beta=0
    gamma=5

    new_image = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    plt.imshow(cv.LUT(new_image, lookUpTable)) 
    """

    def _preparation_frame(self, frame):
        frame = cv.resize(frame, self.IMAGE_SIZE, interpolation = cv.INTER_AREA)
        frame = self.__change_brithness_contrast_balanse(frame, beta=0, alpha=1, gamma=5)
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, -1).repeat(3, -1)
        frame = frame / 255
        frame = frame.astype('float32')
        return frame
    
    def _predict_gest(self, frame) -> int: 
        prepareated_frame = self._preparation_frame(frame)
        
        tensor_frame = torchvision.transforms.ToTensor()(prepareated_frame)
        output = self.gest_model(tensor_frame[None, :].to(self.device))
        # with torch.no_grad():
        #     probabilites = torch.nn.Softmax()(output)
        
        # indx = torch.max(probabilites, dim=1).indices.item()
        indx = torch.max(output, dim=1).indices.item()
        #TODO: Использовать порог для обработки маловероятных событий. 
        # Посмотреть в отладчике вероятность события, когда жест показан
        return indx
         
    def handling(self, gest_id, frame):
        text = str(GestAppNN.INDX2LABEL[gest_id]) if gest_id else 'Unknow gest'
        position = (10, 50)
        font_family = cv.FONT_HERSHEY_SIMPLEX
        font_size = 2
        font_colot = (255, 255, 0)
        
        frame = cv.putText(frame, text, position, font_family, font_size, font_colot)
        return frame
        
    def _handling_frame(self, frame):
        try:
            if self._is_person(frame):
                gest_id = self._predict_gest(frame)
                handled_frame = self.handling(gest_id, frame)
                # if not handled_frame:
                #     print('stop')
                    
                return handled_frame
            return None
        except Exception as e:
            print(e)
            
    def run(self):
        cap = None
        try:
            cap = self._open_cam()
            while True:
                ret, frame = cap.read()
                
                # if frame is read correctly ret is True
                if not ret:
                    raise StreamEndException("Can't receive frame (stream end?). Exiting ...")
                    
                # Our operations on the frame come here
                handled_frame = self._handling_frame(frame.copy())
                # Display the resulting frame
                if handled_frame is not None:
                    cv.imshow('frame', handled_frame)
                else:
                    print("This's is not a human face")
                    cv.imshow('frame', frame)
                    
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
    app = GestApp(Path('GestApp/Models/inception_98.00'))
    app.IMAGE_SIZE = (299, 299) # for inception block
    app.run()
    
