# Server Scripts

* [**Introduction**](#introduction)
* [**GRPC Servers**](#grpc_server)
  * [**Face Analytics Inference Server (GRPC)**](#faceGRPCServer)
  * [**Face Analytics Inference Client (GRPC)**](#faceGRPCClient)
* [**REST API Servers**](#rest_server)
  * [**Face Analytics Inference Server (REST API)**](#faceRestServer)
  * [**Face Analytics Inference Client (REST API)**](#faceRestClient)


## <a name="introduction">Introduction

Server scripts are used to power server client architecture. Server scripts are responsible for inference handling and client identification. Server architecture makes it possible to load trained resources automatically for inference, and listens for client for request. Server client architecture communication is based on GRPC & REST API.

## <a name="grpc_server">GRPC Servers

### <a name="faceGRPCServer">Face Analytics Inference Server (GRPC)

Face Analytics inference server is responsible of batched inference, results computation and handling of clients. This [script][grs] take following environment variables as input:

```bash
export targetWidth=640                                             # Face Detection image width
export targetHeight=640                                            # Face Detection image height
export landmarkWidth=160                                           # Face Landmarks image width
export landmarkHeight=160                                          # Face Landmarks image height
export recognitionWidth=112                                        # Face Recognition image width
export recognitionHeight=112                                       # Face Recognition image height
export spoofHeight=128                                             # Face Anti-Spoofing image height
export spoofWidth=128                                              # Face Anti-Spoofing image width
export invalidThreshold=0.75                                       # Face invalid pose threshold
export facesTemplatesBucket=facebook                               # Faces repository as templates
export detectionModel=/face_detect_Nx3x640x640.onnx                # Face Detection model absolute path
export landmarkModel=/face_landmark_Nx3x160x160.onnx               # Face Landmarks model absolute path
export recognitionModel=/face_recognition_Nx3x112x112.onnx         # Face Recognition model absolute path
export spoofingModel=/anti-spoof_1x3x128x128.onnx                  # Anti-Spoofing model absolute path
export bucketAccessKey=abc                                         # MINIO access key
export bucketSecretKey=abc                                         # MINIO secret key
export secureBucket=false                                          # MINIO bucket https enabled or not
export bucketUrl=facebuckets:9000                                  # MINIO bucket API URL
export serverIp=[::]:8080                                          # GRPC server IP:Port
export messageLength=1000000000                                    # GRPC naximum message length
export numWorkers=1                                                # GRPC number of workers
```

### <a name="faceGRPCClient">Face Analytics Inference Client (GRPC)

Client scripts are used to request images for inference to server. In simple words, they send batch of images to inference server, and return results after process. Usage for this [script][grc] is given as following:

```python
from inferenceGRPCClient import *

face_analytics_inference_server_ip = '172.17.0.2:8080'

img1 = ocv.imread('some image path')
img2 = ocv.imread('some image path')

img_batch = [img1, img2, ...]
detectionThreshold = [thres1, thres2, ...]
nmsThreshold = [nms1, nms2, ...]
boxCushion = [bCu1, bCu2, ...]
smallFacePercent = [sp1, sp2, ...]
recognitionThreshold = [rt1, rt2, ...]
spoofingThreshold = [spo1, spo2, ...]
trackerFrames = [tr1, tr2, ...]

target_server = faceClient(face_analytics_inference_server_ip)

results = target_server(
                img_batch,
                detectionThreshold,
                nmsThreshold,
                boxCushion,
                smallFacePercent,
                recognitionThreshold,
                spoofingThreshold,
                trackerFrames
            )
```

## <a name="rest_server">REST API Servers

### <a name="faceRESTServer">Face Analytics Inference Server (REST API)

Face Analytics inference server is responsible of batched inference, results computation and handling of clients. This [script][ins] take following environment variables as input:

```bash
export grpcServerIP=facegrpcinference:8080      # Face Analytics GRPC inference server
export imageWidth=640                           # Target image width
export imageHeight=640                          # Target image height
export inferencePort=8080                       # REST API inference port
export apiRoute=/face                           # REST API inference route
```

### <a name="faceRESTClient">Face Analytics Inference Client (REST API)

Client scripts are used to request images for inference to server. In simple words, they send batch of images to inference server, and return results after process. Usage for this [script][inc] is given as following:

```python
from inferenceRESTClient import *

face_analytics_inference_server_ip = 'http://172.17.0.2:8080/face'
clientID = 'Your-ID'

img1 = ocv.imread('some image path')
img2 = ocv.imread('some image path')

img_batch = [img1, img2, ...]
detectionThreshold = [thres1, thres2, ...]
nmsThreshold = [nms1, nms2, ...]
boxCushion = [bCu1, bCu2, ...]
smallFacePercent = [sp1, sp2, ...]
recognitionThreshold = [rt1, rt2, ...]
spoofingThreshold = [spo1, spo2, ...]
trackerFrames = [tr1, tr2, ...]

target_server = Detector(face_analytics_inference_server_ip, clientID)

results = target_server(
                img_batch,
                detectionThreshold,
                nmsThreshold,
                boxCushion,
                smallFacePercent,
                recognitionThreshold,
                spoofingThreshold,
                trackerFrames
            )
```

[grs]: ./faceServer.py
[grc]: ./inferenceGRPCClient.py
[ins]: ./inferenceRESTServer.py
[inc]: ./inferenceGRPCClient.py
