# Computer Pointer Controller

Computer Pointer Controller allows user to control the mouse movements with their eye-gaze which will be captured through a webcam or even a video by using OpenVINO Toolkit along with its Pre-Trained Models which helps to deploy AI at Edge. This project will run using multiple models in the same device and coordinate the flow of data between models.

## Project Set Up and Installation
To get up to speed on running this project, you'll need to setup your local environment. Here are the main things to do:
* Download and install the [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html).
* The models used in running the project, * already downloaded in the intel folder.
* i   Face Detection [Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* ii  Facial Landmarks Detection [Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* iii Head Pose Estimation [Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* iv  Gaze Estimation [Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Demo
#### Step 1
- Clone the repository: Open a new terminal and run the following command:-
- cd <path_to_project_directory>/src

#### Step 2
- source the openvino environment 
- source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
Also ensure to check setup.sh

#### Step 3
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

default value for probability threshold of all models is 0.5


### Directory Structure
* bin folder contains the media files

* src folder contains python files of the app

  - main.py : Main python script to run the app
  - model.p : contains python class to handle all models pre-processing.
  - face_detection.py : Face Detection inference code
  - facial_landmarks_detection.py : Landmark Detection inference code
  - gaze_estimation.py : Gaze Estimation inference code
  - head_pose_estimation.py : Head Pose Estimation inference code
  - input_feeder.py : video input selection related code
  - mouse_controller.py : Mouse Control related activities.
  - README.md: Project Readme file which you're currently reading.
  - requirements.txt: All the dependencies of the project are listed there

  - setup.sh: one shot execution script that covers all the prerequisites of the project.
  - v_tune.sh: script to use to check for hotspots in code using intel's Vtune Amplifier.




## Benchmarks
The benchmark result of running my models on Intel(R) Core(TM) i7-4510U CPU @ 2.00GHz
using Intel's DL [workbench](https://docs.openvinotoolkit.org/latest/workbench_docs_Workbench_DG_Install_from_Docker_Hub.html) with multiple model precisions are :
* Face Detection Model

Precision | Latency (ms) | Throughput (fps)
--------- | ------- | ---------
FP32-FP16-Int8 | 94.91 | 30.16

* Facial Landmark Regression Model

Precision | Latency (ms) | Throughput (fps)
--------- | ------- | ---------
FP32 | 0.41 | 2249.26
FP16 | 0.40 | 2232.52
FP16-Int8 | 0.44 | 2152.11

The FP16 precision gives the best latency and FP32 gives the best throughput. In a situation where speed is of utmost importance, 0.01(ms) will matter, else we can trade off latency for throughput. 

* Head-Pose Estimation Model

Precision | Latency (ms) | Throughput (fps)
--------- | ------- | ---------
FP32 | 2.1 | 459.66
FP16 | 2.09 | 462.79
FP16-Int8 | 1.59 | 602.36

The FP16-Int8 precision gives the best latency and the best throughput. It is the most situable for this hardware. 

* Gaze Estimation Model

Precision | Latency (ms) | Throughput (fps)
--------- | ------- | ---------
FP32 | 2.61 | 371.35
FP16 | 2.59 | 376.22
FP16-Int8 | 2.00 | 487.23

The FP16-Int8 precision gives the best latency and the best throughput. It is the most situable for this hardware. 

#### Considering overall model performances, the Face Detection model has the heighest latency and gives the lowest throughput.

## Results

From the above benchmark results obtained, faster inference is obtaned using less precision model. by reducing the precision, the usage of memory is less and its less computationally expensive when compared to higher precision models.

## Stand Out Suggestions

### Async Inference
Using the start_async method will use the all cores of CPU and improve performance with threading which is the  ability to perform multiple inference at the same time compared to synchronous infer method. In synchrounous inference, the inference request need to be waiting until the other inference request executed therefore, it is more suitable to use async inference in this project

### Edge Cases
* This project only works better when just one person has been detected in the frame. In real conditions, if you use webcam, we should deal with detected multiple persons. To deal with condition, If there are Multiple faces detected in the frame then the model takes the first detected face for control of the mouse pointer.

* Some situations inference may break such as when the mouse moves to the corner of the frame.

