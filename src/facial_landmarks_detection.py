import os
import sys
import cv2
import logging as log
import time
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from model import Model


class Facial_Landmark_Detection(Model):
    '''
    Class for the Facial landmark Model.
    '''

    def preprocess_input(self, image):
        '''
        preprocesses inputs before feeding into the model
        '''
        try:
            self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)
        except Exception:
            log.error('Error occured while preprocessing the inputs | Landmark-Detection')
        return self.image

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

            l_eye, r_eye = self.preprocess_output(outputs, image)
            return l_eye, r_eye, inference_time

        except Exception:
            log.error('No prediction was found | Facial-Landmark-Detection Model')
        
        
    def preprocess_output(self, outputs, image):
        '''
        post processeses the output of the predictictions given by the model 
        '''
        outputs = outputs[0]
        # get coordinates for the right eye

        r_eye_xmin = int(outputs[2][0][0] * image.shape[1])- 10
        r_eye_ymin = int(outputs[3][0][0] * image.shape[0])- 10
        r_eye_ymax = int(outputs[3][0][0] * image.shape[0])+ 10
        r_eye_xmax = int(outputs[2][0][0] * image.shape[1])+ 10
        r_eye_img =  image[r_eye_ymin:r_eye_ymax,r_eye_xmin:r_eye_xmax]

        # get coordinates for the lef eye

        l_eye_xmin = int(outputs[0][0][0] * image.shape[1])- 10
        l_eye_ymin = int(outputs[1][0][0] * image.shape[0])- 10
        l_eye_ymax = int(outputs[1][0][0] * image.shape[0])+ 10
        l_eye_xmax = int(outputs[0][0][0] * image.shape[1])+ 10
        l_eye_img =  image[l_eye_ymin:l_eye_ymax,l_eye_xmin:l_eye_xmax]

        try:
            if 'facial_landmark_detection' in self.visual_flag:
                eyes = [[l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax],[r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax]]
                cv2.rectangle(image, (eyes[0][0]-10, eyes[0][1]-10), (eyes[0][2]+10, eyes[0][3]+10), (64, 255, 128), 3  )
                cv2.rectangle(image, (eyes[1][0]-10, eyes[1][1]-10), (eyes[1][2]+10, eyes[1][3]+10), (64, 255, 128), 3  )
            
        except IndexError:
            log.error('eyes were undetected, index error | Facial-Landmark Detection')
            exit()
        return l_eye_img, r_eye_img
        