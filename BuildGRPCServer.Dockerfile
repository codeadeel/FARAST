# syntax=docker/dockerfile:1
#
# =================================================================
# | FACE DETECTION, RECOGNITION & TRACKING INFERENCE SERVER (GRPC)|
# =================================================================
# 
# This Dockerfile is used to build model inference server for GRPC based Face Detection, Recogntion & Tracking.
# 
# Quick Command to Build Inference Server
# =======================================
# docker build -t face:grpcserver -f BuildGRPCServer.Dockerfile .
#
# Quick Command to Run Inference Server
# =====================================
# docker run --rm -it --gpus all \
#     -e targetWidth=640                                        # [ Optional : Target image width for detection inference ] \
#     -e targetHeight=640                                       # [ Optional : Target image height for detection inference ] \
#     -e landmarkWidth=160                                      # [ Optional : Target image width for landmarks inference ] \
#     -e landmarkHeight=160                                     # [ Optional : Target image height for landmarks inference ] \
#     -e recognitionWidth=112                                   # [ Optional : Target image width for recognition inference ] \
#     -e recognitionHeight=112                                  # [ Optional : Target image height for recognition inference ] \
#     -e spoofHeight=128                                        # [ Optional : Target image width for anti-spoofing ] \
#     -e spoofWidth=128                                         # [ Optional : Target image height for anti-spoofing ] \
#     -e invalidThreshold=0.75                                  # [ Optional : Invalid pose threshold ] \
#     -e facesTemplatesBucket=facebook                          # [ Optional : Faces templates repository ] \
#     -e detectionModel=/face_detect_Nx3x640x640.onnx           # [ Optional : Face detection model address ] \
#     -e landmarkModel=/face_landmark_Nx3x160x160.onnx          # [ Optional : Face landmark model address ] \
#     -e recognitionModel=/face_recognition_Nx3x112x112.onnx    # [ Optional : Face recognition model address ] \
#     -e spoofingModel=/anti-spoof_1x3x128x128.onnx             # [ Optional : Face anti-spoofing model address ] \
#     -e bucketAccessKey=abc                                    # [ Required : MINIO access key ] \
#     -e bucketSecretKey=abc                                    # [ Required : MINIO secret key ] \
#     -e secureBucket=false                                     # [ Optional : MINIO bucket https enabled or not ] \
#     -e bucketUrl='http://172.17.0.5:8009'                     # [ Required : MINIO bucket api address ] \
#     -e serverIp=[::]:8080                                     # [ Optional : Server IP for GRPC Server ] \
#     -e messageLength=1000000000                               # [ Optional : Message length for GRPC Server ] \
#     -e numWorkers=1                                           # [ Optional : Number of workers for GRPC Server ] \
#     -v [ Required: Your Trained ONNX Model for Face Detection ]:/face_detect_Nx3x640x640.onnx \
#     -v [ Required: Your Trained ONNX Model for Face Landmarks ]:/face_landmark_Nx3x160x160.onnx \
#     -v [ Required: Your Trained ONNX Model for Face Recognition ]:/face_recognition_Nx3x112x112.onnx \
#     -v [ Required: Your Trained ONNX Model for Face Anti-Spoofing ]:/anti-spoof_1x3x128x128.onnx:ro \
#     face:grpcserver
#
# Main Build Script
# =================
#
# Pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime Image from DOCKER-HUB
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# Install Necessary Packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing ; apt-get install -y build-essential
RUN pip3 install --no-cache-dir opencv-python-headless==4.8.0.74 opencv-contrib-python-headless==4.8.0.74 minio==7.1.15 onnxruntime-gpu==1.15.1 grpcio==1.56.2 dlib==19.24.2
# Copy Server Files for Execution
COPY ./Server/faceCommunication_pb2.py ./faceCommunication_pb2.py
COPY ./Server/faceCommunication_pb2_grpc.py ./faceCommunication_pb2_grpc.py
COPY ./Server/faceTools.py ./faceTools.py
COPY ./Server/faceServer.py ./faceServer.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./faceServer.py
ENTRYPOINT [ "./faceServer.py" ]
