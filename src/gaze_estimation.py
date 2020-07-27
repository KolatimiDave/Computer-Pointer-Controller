import math
import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from model import Model

class Gaze_Estimation(Model):
    '''
    Class for the Gaze estimation Model.
    '''
    def predict(self, left_eye_image, right_eye_image, head_pose_angles_coords,image ):
        '''
        performs predictions given to the model
        '''
        try:

            left_eye_image = self.preprocess_input(left_eye_image)
            right_eye_image = self.preprocess_input(right_eye_image)

            inference_time = time.time()
            self.exec_net.start_async(0, inputs={'left_eye_image':left_eye_image, 'right_eye_image':right_eye_image, 'head_pose_angles': head_pose_angles_coords})
            if self.exec_net.requests[0].wait(-1) == 0:
                outputs = self.exec_net.requests[0].outputs
                inference_time = time.time() - inference_time

                mouse_coord, gaze_vector = self.preprocess_output(outputs, head_pose_angles_coords, image)

        except Exception:
            log.error('error occured while perfroming gaze prediction')

        return mouse_coord, gaze_vector, inference_time
   

    def preprocess_input(self, image):
        '''
        preprocesses inputs before feeding into the model
        '''
        try:
            self.image = cv2.resize(image, (60, 60))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)

        except Exception:
            log.error('Error occured while preprocessing the inputs | Gaze-estimation')
        return self.image

    def preprocess_output(self, outputs, head_pose_angles, image):
        '''
        post processeses the output of the predictions given by the model 
        '''
        gaze_vector = outputs[self.output_blob][0]
        mouse_coord = (0,0)
        try:
            angle_r_fc = head_pose_angles[2]
            sin_r = math.sin(angle_r_fc * math.pi / 180)
            cos_r = math.cos(angle_r_fc * math.pi / 180)
            x = gaze_vector[0] * cos_r + gaze_vector[1] * sin_r
            y = -gaze_vector[0] * sin_r + gaze_vector[1] * cos_r
            mouse_coord = (x, y)
            
            if 'gaze_estimation' in self.visual_flag:
                cv2.putText(image, "gazevector_x = {:.2f}, gazevector_y = {:.2f}, gazevector_z = {:.2f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 50, 150), 3)
            return mouse_coord, gaze_vector
        except Exception :
            log.error('error occured while processing model output | gaze estimation')