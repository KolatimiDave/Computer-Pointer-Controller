source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6
python3.6 ./src/main.py -fdm /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -flm /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hpm /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -gem /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -v /home/kolatimi/Desktop/project3/project3/starter/bin/demo.mp4 