# syntax=docker/dockerfile:1
#
# ================================
# | FACE DETECTION MODEL TRAINER |
# ================================
#
# This Dockerfile is used to build Face Detection Model Trainer
# 
# Quick Command to Build Model Trainer:
# =====================================
# docker build -t face:trainer -f trainer.dockerfile --no-cache .
#
# Quick Command to Run Model Trainer:
# ===================================
# docker run --rm -it --gpus all \
#            -v [ Required : Face Detection Dataset ]:/data \
#            -v [ Optional : Model Trainer Output ]:/output \
#            -e batchSize=10
#            -e epochs=10
#            -e workers=8
#            face:trainer
#
# Face Dataset Structure:
# =======================
# /data
#   /images
#     /train
#       - 000000.jpg
#       - 000001.jpg
#       - ...
#     /val
#       - 000000.jpg
#       - 000001.jpg
#       - ...
#   /labels
#     /train
#       - 000000.txt    >> Format: [ Class Number, Cx, Cy, Width, Height ]
#       - 000001.txt
#       - ...
#     /val
#       - 000000.txt
#       - 000001.txt
#       - ...
# Pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel Image from DOCKER-HUB
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
# Install Necessary Packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing ; apt-get -y upgrade
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0 libgtk-3-dev
RUN pip3 install super-gradients==3.1.2
# Copy Trainer Files for Execution
RUN mkdir -p /root/.cache/torch/hub/checkpoints
COPY ./Resources/yolo_nas_s_coco.pth /root/.cache/torch/hub/checkpoints/yolo_nas_s_coco.pth
WORKDIR /workspace
COPY ./Trainer/trainer.py ./trainer.py
# Set Permissions & Create Execution Entrypoint
RUN chmod 777 ./trainer.py
ENTRYPOINT ["./trainer.py"]
