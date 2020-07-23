# Computer Pointer Controller

Computer Pointer Controller allows user to control the mouse movements with their eye-gaze which will be captured through a webcam or even a video by using OpenVINO Toolkit along with its Pre-Trained Models which helps to deploy AI at Edge. This project will run using multiple models in the same device and coordinate the flow of data between models.

## Project Set Up and Installation
To get up to speed on running this project, you'll need to setup your local environment. Here are the main things to do:
* Download and install the OpenVINO Toolkit[https://docs.openvinotoolkit.org/latest/index.html].
* The models used in running the project *Already downloaded in the intel folder.
i   Face Detection [Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
ii  Facial Landmarks Detection [Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
iii Head Pose Estimation [Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
iv  Gaze Estimation [Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Demo
* Step 1
- Clone the repository: Open a new terminal and run the following command:-
- cd <path_to_project_directory>/src

* Step 2
Now, run the following command to run our application
* python3.6 main.py -fdm <path_to_project_directory>/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -flm <path_to_project_directory>/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hpm <path_to_project_directory>/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -gem <path_to_project_directory>/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -v <path_to_project_directory>/bin/demo.mp4



## Documentation
* The python main.py -h command displays the commands which are supported by
-h, --help show this help message and exit

*   -fdm, face_detection_model, Path to .xml file of pretrained face detection model
*    -hpm, headpose_model, Path to .xml file of pretrained head pose estimation model
*    -flm, facial_landmark_model, Path to .xml file of pretrained facial landmark model
*    -gem, gaze_estimation_model, Path to .xml file of pretrained gaze estimation model
*    -d, device, specify target device to infer on, device can be: CPU, GPU, FPGA or MYRIAD. Default is CPU
*    -v, video_input, enter path to video file or cam to use webcam
*    -l, cpu_extension, specify path to cpu extension
*    -fdp, face_detection_prob_threshold, specify probaility threshold for the face detection model
*    -hpp, headpose_prob_threshold, specify probaility threshold for the head pose model
*    -flp, facial_landmark_prob_threshold, specify probaility threshold for the facial landmark model
*    -gep, gaze_estimation_prob_threshold, specify probaility threshold for the gaze estimation model
        

## Benchmarks
* The benchmark result of running my model on CPU with multiple model precisions are :

- INT8:

The total model loading time is : 0.838sec
The total inference time is : 0.699sec
The total FPS is : 1.43fps
- FP16:

The total model loading time is : 0.731sec
The total inference time is : 0.668sec
The total FPS is : 1.498fps

- FP32:

The total model loading time is : 0.838sec
The total inference time is : 0.665sec
The total FPS is : 1.505fps


## Results

## Stand Out Suggestions
I improved my model inference time by using multiple precisions of the models.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
