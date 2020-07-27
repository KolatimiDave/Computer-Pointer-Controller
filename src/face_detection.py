import cv2
import logging as log
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
from model import Model


class Face_Detection(Model):
    '''
    Class for the Face Detection Model.
    '''
    def predict(self, image):
        '''
        performs the predictions given to the model
        '''
        try:
            preprocessed_image = self.preprocess_input(image)
            
            inference_time = time.time()
            results = self.exec_net.infer({self.input_blob:preprocessed_image})
            outputs = results[self.output_blob]
            inference_time = time.time() - inference_time
        
            self.points, self.image = self.preprocess_output(outputs,image)
            try:
                cropped_image = image[self.points[0][1]: self.points[0][3], self.points[0][0]: self.points[0][2]]
            except IndexError:
                log.error('Index out of frame, Unable to detect right eye now')
                exit()
            return image, cropped_image, outputs, inference_time

        except IndentationError:
            log.error('No Face has been detected | Face-Detection Model ')
            exit()
    

    def preprocess_input(self, image):
        '''
        preprocesses inputs before feeding into the model
        '''
        try:
            self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)
            
        except Exception :
            log.error('Error occured while preprocessing the inputs | Face-Detection-Model')
        return self.image

    def preprocess_output(self, outputs, image):
        '''
        post processeses the output of the predictions given by the model 
        '''
        points = []
        for box in outputs[0][0]:
            confidence = box[2]
            if confidence > self.prob_thresh:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                points.append((xmin,ymin,xmax,ymax))
                if 'face_detection' in self.visual_flag:
                    image = cv2.rectangle(image, (xmin, ymin),(xmax, ymax), (10,20,200), 1)
        return points, image