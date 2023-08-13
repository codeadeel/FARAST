# syntax=docker/dockerfile:1
#
# ==========================================================
# | FACE DETECTION, RECOGNITION & TRACKING FACE ENROLLMENT |
# ==========================================================
# 
# This Dockerfile is used to build face enrollment app for Face Detection, Recognition, Anti-Spoofing & Tracking.
#
# Quick Command to Build Face Enrollment
# ======================================
# docker build -t face:enrollment -f BuildEnrollment.Dockerfile .
#
# Quick Command to Run Face Enrollment
# ====================================
# docker run --rm -it \
#     -e submissionCode=abc                                     # [ Required : Passcode for Data Submission ] \
#     -e facesTemplatesBucket=facebook                          # [ Optional : Faces templates repository ] \
#     -e bucketAccessKey=abc                                    # [ Required : MINIO access key ] \
#     -e bucketSecretKey=abc                                    # [ Required : MINIO secret key ] \
#     -e secureBucket=false                                     # [ Optional : MINIO bucket https enabled or not ] \
#     -e bucketUrl='http://172.17.0.5:8009'                     # [ Required : MINIO bucket https enabled or not ] \
#     -e serverIP='http://172.17.0.5:8080'                      # [ Required : Face Server Inference API ] \
#     face:enrollment
#
# Main Build Script
# =================
#
# Pull ubuntu:latest Image from Docker-Hub
FROM ubuntu:latest
# Install Necessary Packages
RUN apt-get update ; apt-get install -y --fix-missing ; apt-get install -y python3 python3-pip libgl1-mesa-dev libglib2.0-0 libgtk-3-dev
RUN pip3 install opencv-python-headless==4.7.0.68 streamlit==1.21.0 streamlit-webrtc==0.45.1 altair==4.2.2 grpcio==1.56.2 bounding-box==0.1.3 minio==7.1.15
# Copy Resources to Respective Directories
RUN mkdir /root/.streamlit
COPY ./demoApp/enrollmentConfig.toml /root/.streamlit/config.toml
WORKDIR /home
COPY ./Server/faceCommunication_pb2.py ./faceCommunication_pb2.py
COPY ./Server/faceCommunication_pb2_grpc.py ./faceCommunication_pb2_grpc.py
COPY ./Server/inferenceGRPCClient.py ./inferenceGRPCClient.py
COPY ./demoApp/logo.png ./logo.png
COPY ./demoApp/faceEnroll.py ./demo.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./demo.py
ENTRYPOINT ["streamlit", "run", "./demo.py"]
