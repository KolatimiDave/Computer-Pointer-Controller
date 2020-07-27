import time
import cv2
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore
from model import Model


class Head_Pose_Estimation(Model):
    '''
    Class for the HeadPose Estimation Model.
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
            log.error('Error occured while preprocessing the inputs | Head-Pose-Estimation')
        return self.image



    def predict(self, cropped_image, frame):
        '''
        performs predictions on image given to the model
        '''
        try:
            preprocessed_img = self.preprocess_input(cropped_image)
            inference_time = time.time()

            self.exec_net.start_async(0, inputs={self.input_blob: preprocessed_img})
            if self.exec_net.requests[0].wait(-1) == 0:
                outputs = self.exec_net.requests[0].outputs
                inference_time = time.time() - inference_time

                results = self.preprocess_output(outputs, frame)
            
        except Exception:
            log.error('No prediction was found | Head-Pose-Estimation Model')
            exit()
        return results, inference_time
        
    def preprocess_output(self, outputs, frame):
        '''
        post process the results from the model
        '''
        try:
            results = []
            results.append(outputs['angle_y_fc'][0][0])
            results.append(outputs['angle_p_fc'][0][0])
            results.append(outputs['angle_r_fc'][0][0])
            if 'head_pose_estimation' in self.visual_flag:
                cv2.putText(frame, 'Pose angles in degree: yaw = {:.2f}, pitch = {:.2f}, roll = {:.2f}'.format(results[0], results[1], results[2]), (15, 105), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 3)

        except Exception:
            log.error('error occured while processing model output | gaze estimation')

        return results                