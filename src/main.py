from argparse import ArgumentParser
import os
import numpy as np
import logging as log
import time
import cv2
from mouse_controller import MouseController
from input_feeder import InputFeeder
from face_detection import Face_Detection
from head_pose_estimation import Head_Pose_Estimation
from facial_landmarks_detection import Facial_Landmark_Detection
from gaze_estimation import Gaze_Estimation


def Initailize_model(model_class, model_name, device, extensions, prob_thresh, visual_flag):
    '''
    Load models and checks visual flag 
    '''
    model_load_time = time.time()
    if model_class.__name__.lower() in visual_flag:
        model = model_class(model_name = model_name, device = device, extensions = extensions, prob_thresh = prob_thresh, visual_flag = visual_flag)
    else:
        model = model_class(model_name = model_name, device = device, extensions = extensions, prob_thresh = prob_thresh)
    model.load_model()
    model_load_time = time.time() - model_load_time
    return model, model_load_time
    

def main(args):
    video_input = args.video_input
    video_type = None
    feed = None

    if video_input != 'cam':
        if not os.path.isfile(video_input):
            log.error('path to video file does not exist')
            exit(1)
        video_type = 'video'
        feed = InputFeeder(input_type = video_type, input_file = video_input)
    elif video_input == 'cam':
        video_type = 'cam'
        feed = InputFeeder(input_type = video_type, input_file = video_input)
    else:
        log.error('Please enter either path to a video file or cam for web camera')
  

    Face_Detector, face_detector_model_load_time = Initailize_model(model_class =  Face_Detection, model_name = args.face_detection_model, device = args.device, extensions = args.cpu_extension, prob_thresh = args.face_detection_prob_threshold, visual_flag = args.visual_flag)
    Head_Pose_Estimator, head_pose_estimator_model_load_time = Initailize_model(model_class =  Head_Pose_Estimation, model_name = args.headpose_model, device = args.device, extensions = args.cpu_extension, prob_thresh = args.headpose_prob_threshold, visual_flag = args.visual_flag)
    Facial_Landmark_Detector, facial_landmark_detector_model_load_time = Initailize_model(model_class =  Facial_Landmark_Detection, model_name = args.facial_landmark_model, device = args.device, extensions = args.cpu_extension, prob_thresh = args.facial_landmark_prob_threshold, visual_flag = args.visual_flag)
    Gaze_Estimator, gaze_estimator_model_load_time = Initailize_model(model_class =  Gaze_Estimation, model_name = args.gaze_estimation_model, device = args.device, extensions = args.cpu_extension, prob_thresh = args.gaze_estimation_prob_threshold, visual_flag = args.visual_flag)
    
    model_load_time_dict = {
        'face_detector': face_detector_model_load_time,
        'facial_landmark': facial_landmark_detector_model_load_time,
        'headpose': head_pose_estimator_model_load_time,
        'gaze_estimator': gaze_estimator_model_load_time
    }
    max_load_time = max(model_load_time_dict, key=model_load_time_dict.get)

    frame_count = 0
    feed.load_data()
    fps = feed.calculate_fps()

    for flag, frame in feed.next_batch():
        if frame is None:
            log.error('could not read video input')
            exit()
        if not flag:
            break
        frame_count += 1

        total_inference_time = time.time()

        # Input frames are fed to models for inference.
        # Outputs from multiple models are fed consecutively to other models.

        frame, face_crop, detections, face_detector_inference_time = Face_Detector.predict(frame)
        left_eye, right_eye, facial_landmark_detector_inference_time = Facial_Landmark_Detector.predict(face_crop)
        head_pose_result, head_pose_estimator_inference_time = Head_Pose_Estimator.predict(face_crop, frame)
        mouse_coord, gaze_vector, gaze_estimator_inference_load_time = Gaze_Estimator.predict(left_eye, right_eye, head_pose_result, frame)

        total_inference_time = time.time() - total_inference_time
        total_inf_time_message = 'Inference time: {:.3f}ms'\
            .format(total_inference_time)
        cv2.putText(frame, total_inf_time_message, (15,70), cv2.FONT_HERSHEY_COMPLEX, 1, (15, 70, 157), 3)
        
        if frame is not None:
            cv2.imshow('frame', frame)

        mouseController = MouseController('medium','moderate')
        if frame_count % 4 == 0:
            mouseController.move(mouse_coord[0], mouse_coord[1])

        inference_time_dict = {
        'face_detector': face_detector_inference_time,
        'facial_landmark': facial_landmark_detector_inference_time,
        'headpose': head_pose_estimator_inference_time,
        'gaze_estimator': gaze_estimator_inference_load_time
        }
        max_inference_time = max(inference_time_dict, key=inference_time_dict.get)

    cv2.destroyAllWindows()
    feed.close()

    print('Total number of frames per second is {:.3f} fps'.format(fps))
    print('{}_model has the highest load time of {:.3f} seconds'.format(max_load_time, model_load_time_dict.get(max_load_time)))
    print('{}_model has the highest inference time of {:.3f} seconds'.format(max_inference_time, inference_time_dict.get(max_inference_time)))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-fdm', '--face_detection_model', required = True, type = str, help = 'Path to .xml file of pretrained face detection model')
    parser.add_argument('-hpm', '--headpose_model', required = True, type = str, help = 'Path to .xml file of pretrained head pose estimation model')
    parser.add_argument('-flm', '--facial_landmark_model', required = True, type = str, help = 'Path to .xml file of pretrained facial landmark model')
    parser.add_argument('-gem', '--gaze_estimation_model', required = True, type = str, help = 'Path to .xml file of pretrained gaze estimation model')
    parser.add_argument('-d', '--device', type = str, default = 'CPU', required = False, help = 'specify target device to infer on, device can be: CPU, GPU, FPGA or MYRIAD. Default is CPU')
    parser.add_argument('-v', '--video_input', required = True, type = str, help = 'enter path to video file or cam to use webcam')
    parser.add_argument('-l', '--cpu_extension', type = str, default = None, required = False, help = 'specify path to cpu extension')
    parser.add_argument('-fdp', '--face_detection_prob_threshold', required = False, type = float, default = 0.7, help = 'specify probaility threshold for the face detection model')
    parser.add_argument('-hpp', '--headpose_prob_threshold', required = False, type = float, default = 0.7, help = 'specify probaility threshold for the head pose model')
    parser.add_argument('-flp', '--facial_landmark_prob_threshold', required = False, type = float, default = 0.7, help = 'specify probaility threshold for the facial landmark model')
    parser.add_argument('-gep', '--gaze_estimation_prob_threshold', required = False, type = float, default = 0.7, help = 'specify probaility threshold for the gaze estimation model')
    parser.add_argument('-flags', '--visual_flag', required = False, nargs = '+', default = [], help = 'To visualize the output of the models. For face detection enter face_detection, for facial landmark detetcion, enter facial_landmark_detection, for head pose estimation enter head_pose_estimation, for gaze estimation enter gaze_estimation' )

    args = parser.parse_args()
    main(args)