python3 -m venv venv
source venv/bin/activate
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6 
pip install -r requirements.txt
mkdir models
python3.6 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 
python3.6 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001
python3.6 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009
python3.6 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002
