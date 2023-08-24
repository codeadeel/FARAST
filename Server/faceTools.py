#!/usr/bin/env python3

"""
MODEL INFERENCE TOOLS
=====================

Following program is provides the tools for subject inference
"""

# %%
# Importing Libraries
import os
import string
from io import BytesIO
import numpy as np
import torch
import torchvision as tv
import dlib
import cv2 as ocv
from PIL import Image
from minio import Minio
import onnxruntime as ort

# %%
# Image Definition for Inference Model
class imgData:
    def __init__(
        self,
        dat: np.array,
        trackerId: string,
        detectionThreshold: float = 0.75,
        nmsThreshold: float = 0.5,
        boxCushion: int = 0,
        smallFaceAreaPercent: int = 1,
        recognitionThreshold: float = 0.5,
        fakeThres: float = 0.5,
        frames2Track: int = 30
    ) -> None:
        """
        This method initializes the image class, which processes the image according to model requirements

        Arguments
        =========
        dat : OpenCV image for inference
        trackerId : Stream Id, from which image is coming from, for tracking purposes ( Can be random, but should be same for data stream identification)
        detectionThreshold : Bounding box confidence threshold
        nmsThreshold : Non-Max-Suppression threshold in bounding box detection
        boxCushion : Bounding box padding pixels, which are added to bounding box dimensions
        smallFaceAreaPercent : Percentage area of bounding box, lesser than which face is considered as small
        recognitionThreshold : Face recognition threshold
        fakeThres : Face fakeness threshold
        frames2Track : Number of frames of the stream to track
        """
        self.rawData = dat
        self.data = ocv.cvtColor(self.rawData, ocv.COLOR_BGR2RGB)
        self.trackerId = trackerId
        self.detectionThreshold = detectionThreshold
        self.nmsThreshold = nmsThreshold
        self.boxCushion = boxCushion
        self.smallFaceAreaPercentage = smallFaceAreaPercent
        self.recognitionThreshold = recognitionThreshold
        self.fakenessThreshold = fakeThres
        self.trackerFrames = frames2Track
        self.dataShape = self.data.shape
        self.targetWidth = int(os.environ.get('targetWidth', 640))
        self.targetHeight = int(os.environ.get('targetHeight', 640))
        if (self.dataShape == (self.targetWidth, self.targetHeight, 3)):
            self.inputTensor = self.data.transpose(2, 0, 1)
        else:
            self.inputTensor = ocv.resize(self.data, (self.targetWidth, self.targetHeight)).transpose(2, 0, 1)

# Model Inference Class
class Inference:
    def __init__(self) -> None:
        """
        This method initializes model inference class
        """
        self.targetWidth = int(os.environ.get('targetWidth', 640))
        self.targetHeight = int(os.environ.get('targetHeight', 640))
        self.landmarkWidth = int(os.environ.get('landmarkWidth', 160))
        self.landmarkHeight = int(os.environ.get('landmarkHeight', 160))
        self.recognitionHeight = int(os.environ.get('recognitionHeight', 112))
        self.recognitionWidth = int(os.environ.get('recognitionWidth', 112))
        self.spoofHeight = int(os.environ.get('spoofHeight', 128))
        self.spoofWidth = int(os.environ.get('spoofWidth', 128))
        self.invalidPoseThres = float(os.environ.get('invalidThreshold', '0.75'))
        self.faceBookBucket = os.environ.get('facesTemplatesBucket', 'facebook')
        self.detectionModelAddr = os.environ.get('detectionModel', '/face_detect_Nx3x640x640.onnx')
        self.landmarkModelAddr = os.environ.get('landmarkModel', '/face_landmark_Nx3x160x160.onnx')
        self.recognitionModelAddr = os.environ.get('recognitionModel', '/face_recognition_Nx3x112x112.onnx')
        self.spoofingModelAddr = os.environ.get('spoofingModel', '/anti-spoof_1x3x128x128.onnx')
        try:
            self.detectionSession  = ort.InferenceSession(self.detectionModelAddr, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print('DETECTION SESSION : Yikes! CUDA is not available. Falling back to CPU')
            self.detectionSession  = ort.InferenceSession(self.detectionModelAddr, providers=['CPUExecutionProvider'])
        try:
            self.landmarkSession = ort.InferenceSession(self.landmarkModelAddr, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print('LANDMARK SESSION : Yikes! CUDA is not available. Falling back to CPU')
            self.landmarkSession = ort.InferenceSession(self.landmarkModelAddr, providers=['CPUExecutionProvider'])
        try:
            self.recognitionSession = ort.InferenceSession(self.recognitionModelAddr, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print('RECOGNITION SESSION : Yikes! CUDA is not available. Falling back to CPU')
            self.recognitionSession = ort.InferenceSession(self.recognitionModelAddr, providers=['CPUExecutionProvider'])
        try:
            self.spoofingSession = ort.InferenceSession(self.spoofingModelAddr, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            print('ANTI-SPOOFING SESSION : Yikes! CUDA is not available. Falling back to CPU')
            self.spoofingSession = ort.InferenceSession(self.spoofingModelAddr, providers=['CPUExecutionProvider'])
        self.bucketClient = Minio(
            os.environ.get('bucketUrl', '0.0.0.0:1234'),
            os.environ.get('bucketAccessKey', 'abc'),
            os.environ.get('bucketSecretKey', 'abc'),
            secure = True if os.environ.get('secureBucket', 'false').lower()=='true' else False
        )
        if not self.bucketClient.bucket_exists(self.faceBookBucket):
            self.bucketClient.make_bucket(self.faceBookBucket)
        self.faceBook = list()
        self.faceBookEmbeddings = {'embeddings': list(), 'labels': list()}
    
    def __getPreFaces__(self) -> tuple:
        """
        This method is used to fetch new faces from the storage bucket & generate embeddings & labels

        Outputs:
        ========
        Face embeddings & Labels
        """
        faceBookCheck = [i.object_name for i in self.bucketClient.list_objects(self.faceBookBucket)]
        if len(self.faceBook)!=len(faceBookCheck):
            self.faceBook = list()
            faceBookListRaw = list()
            faceBookListRawShape = list()
            faceBookList = list()
            self.faceBookEmbeddings = {'embeddings': list(), 'labels': list()}
            for i in faceBookCheck:
                self.faceBook.append(i)
                self.faceBookEmbeddings['labels'].append(i.split('@')[0])
                ret = self.bucketClient.get_object(self.faceBookBucket, i)
                ret = np.asarray(Image.open(BytesIO(ret.data)))
                faceBookListRaw.append(ret)
                faceBookListRawShape.append(ret.shape[:2])
                faceBookList.append(
                    ocv.resize(ret, (self.landmarkWidth, self.landmarkHeight)).transpose(2, 0, 1)[np.newaxis, ...]
                )
            faceBookList = np.concatenate(faceBookList)
            rawLandmarks = self.landmarkSession.run(None, {"images": faceBookList.astype(np.float32)})
            rawLandmarks = rawLandmarks[2].reshape([len(rawLandmarks[2]), -1, 2])
            alignedFaces = list()
            for k, sh, orImg in zip(rawLandmarks, faceBookListRawShape, faceBookListRaw):
                k[:, 0] *= sh[1]
                k[:, 1] *= sh[0]
                rightEye = k[37] + ((k[40] - k[37]) / 2)
                leftEye = k[43] + ((k[46] - k[43]) / 2)
                dy = leftEye[1] - rightEye[1]
                dx = leftEye[0] - rightEye[0]
                eyesCenter = np.array([rightEye[0] + (dx / 2), rightEye[1] + (dy / 2)])
                angle = np.degrees(np.arctan2(dy, dx))
                rotMat = ocv.getRotationMatrix2D(eyesCenter, angle, scale=1)
                orImg = ocv.warpAffine(orImg, rotMat, dsize=(sh[1], sh[0]))
                alignedFaces.append(
                    (ocv.resize(orImg, (self.recognitionWidth, self.recognitionHeight)) / 255)[np.newaxis, ...]
                )
            alignedFaces = np.concatenate(alignedFaces)
            self.faceBookEmbeddings['embeddings'] = self.recognitionSession.run(None, {"input_2": alignedFaces.astype(np.float32)})[0]
            return self.faceBookEmbeddings['embeddings'], self.faceBookEmbeddings['labels']
        else:
            return self.faceBookEmbeddings['embeddings'], self.faceBookEmbeddings['labels']
    
    def poseChecker(self, lmks: np.array) -> np.array:
        """
        This method is used to check invalid face pose

        Arguments:
        ==========
        lmks : Available faces lankmarks

        Outputs:
        ========
        Check for if face pose is invalid
        """
        totalxDist = lmks[:, 16, 0] - lmks[:, 2, 0]
        leftDistP = (lmks[:, 31, 0] - lmks[:, 2, 0]) / totalxDist
        rightDistP = (lmks[:, 16, 0] - lmks[:, 31, 0]) / totalxDist
        leftUpDist = lmks[:, 31, 0] - lmks[:, 20, 0]
        rightUpDist = lmks[:, 31, 1] - lmks[:, 25, 1]
        upDist = np.max(np.array([leftUpDist, rightUpDist]), axis=0)
        downDist = lmks[:, 9, 1] - lmks[:, 31, 1]
        totalyDist = downDist + upDist
        upDistP =upDist / totalyDist
        downDistP = downDist / totalyDist
        mapper = np.array([leftDistP, rightDistP, upDistP, downDistP]) >= self.invalidPoseThres
        return np.where(np.sum(mapper, axis=0)==0, False, True)
    
    def __call__(self, imgList: list) -> tuple:
        """
        This method performs inference on the provided image class object pool

        Arguments
        =========
        imgList : Image object pool, specifically designed for model inference

        Outputs
        =======
        finalBBOXsRaw : Bounding boxes as [ Box Probability, x1, y1, x2, y2 ], with respect to full dimension image
        finalRawLandmarks : Face normalized landmarks
        finalFaceProability : Face recognition probability
        finalFaceLabels : Recognized faces labels
        finalSpoofList : Checklist to identify spoofed face
        """
        finalBBOXsRaw = list()
        finalRawLandmarks = list()
        finalFaceProability = list()
        finalFaceLabels = list()
        finalPoseList = list()
        finalSpoofList = list()

        # Face Detection
        inputTensor = np.array([i.inputTensor for i in imgList])
        faceDetections = self.detectionSession.run(None, {"input": inputTensor.astype(np.float32)})

        for bboxes, confidence, img in zip(faceDetections[0], faceDetections[1], imgList):
            tArea = img.dataShape[0] * img.dataShape[1]
            fconf = confidence[confidence[:, 0]>img.detectionThreshold, :]
            fbbox = bboxes[confidence[:, 0]>img.detectionThreshold, :]
            fbbox[:, [0, 2]] /= self.targetWidth
            fbbox[:, [1, 3]] /= self.targetHeight
            fbbox[:, [0, 2]] *= img.dataShape[1]
            fbbox[:, [1, 3]] *= img.dataShape[0]
            idx = tv.ops.nms(torch.from_numpy(fbbox), torch.from_numpy(fconf[:, 0]), img.nmsThreshold).tolist()
            fbbox = fbbox[idx]
            fconf = fconf[idx]
            probWBox = np.concatenate((fconf, fbbox), axis=1)
            probWBox[:, 1:3] -= img.boxCushion
            probWBox[:, 3:5] += img.boxCushion
            probWBox[:, 1:3] = np.where(probWBox[:, 1:3]<0, 0, probWBox[:, 1:3])
            probWBox[:, 3] = np.where(probWBox[:, 3]>img.dataShape[1], img.dataShape[1], probWBox[:, 3])
            probWBox[:, 4] = np.where(probWBox[:, 4]>img.dataShape[0], img.dataShape[0], probWBox[:, 4])
            if len(probWBox)==0:
                finalBBOXsRaw.append([])
                finalRawLandmarks.append([])
                finalFaceProability.append([])
                finalFaceLabels.append([])
                finalPoseList.append([])
                finalSpoofList.append([])
                continue
            else:
                finalBBOXsRaw.append(probWBox.tolist())
            probBArea = (probWBox[:, 3] - probWBox[:, 1]) * (probWBox[:, 4] - probWBox[:, 2])
            probBArea = (probBArea * 100) / tArea

            # Face Cropping & Landmarks Detection
            tempBatch = list()
            for box in probWBox:
                face = img.data[int(box[2]):int(box[4]), int(box[1]):int(box[3]), :]
                tempBatch.append(
                    ocv.resize(face, (self.landmarkWidth, self.landmarkHeight)).transpose(2, 0, 1)
                )
            tempBatch = np.array(tempBatch)
            rawLandmarks = self.landmarkSession.run(None, {"images": tempBatch.astype(np.float32)})
            rawLandmarks = rawLandmarks[2].reshape([len(rawLandmarks[2]), -1, 2])
            finalPoseList.append(self.poseChecker(rawLandmarks))
            finalRawLandmarks.append(rawLandmarks.tolist())

            # Landmarks Mapping, Face Alignment & Resize
            alignedFaces = list()
            spoofOutList = list()
            hei, wi = img.dataShape[:2]
            for k, l in zip(rawLandmarks, probWBox):
                k[:, 0] *= int(l[3] - l[1])
                k[:, 1] *= int(l[4] - l[2])
                k[:, 0] += int(l[1])
                k[:, 1] += int(l[2])
                rightEye = k[37] + ((k[40] - k[37]) / 2)
                leftEye = k[43] + ((k[46] - k[43]) / 2)
                dy = leftEye[1] - rightEye[1]
                dx = leftEye[0] - rightEye[0]
                eyesCenter = np.array([rightEye[0] + (dx / 2), rightEye[1] + (dy / 2)])
                angle = np.degrees(np.arctan2(dy, dx))
                rotMat = ocv.getRotationMatrix2D(eyesCenter, angle, scale=1)
                finalImg = ocv.warpAffine(img.data.copy(), rotMat, dsize=(wi, hei))
                finalImg = finalImg[int(l[2]):int(l[4]), int(l[1]):int(l[3]), :]
                alignedFaces.append(
                    (ocv.resize(finalImg, (self.recognitionWidth, self.recognitionHeight)) / 255)[np.newaxis, ...]
                )
                sR = self.spoofingSession.run(None, {
                    "actual_input_1": ocv.resize(finalImg, (self.spoofWidth, self.spoofHeight)).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
                })[0][0]
                if sR[0]>=img.fakenessThreshold:
                    spoofOutList.append('Real')
                else:
                    spoofOutList.append('Fake')
            alignedFaces = np.concatenate(alignedFaces)
            finalSpoofList.append(spoofOutList)

            # Face Embeddings
            preEmbeddings, preLabels = self.__getPreFaces__()
            if len(preEmbeddings)==0:
                finalFaceLabels.append(['Unknown'] * len(alignedFaces))
                finalFaceProability.append([0] * len(alignedFaces))
            else:
                faceEmbeddings = self.recognitionSession.run(None, {"input_2": alignedFaces.astype(np.float32)})[0]
                faceSim = torch.nn.functional.cosine_similarity(torch.Tensor(faceEmbeddings).unsqueeze(1), torch.Tensor(preEmbeddings), dim=2, eps=1e-6)
                faceMaxProb, faceMaxInd = torch.max(faceSim, axis=1)
                faceRetLab = np.array(preLabels)[faceMaxInd.numpy()]
                faceRetLab = np.where(faceMaxProb.numpy() >= img.recognitionThreshold, faceRetLab, 'Unknown')
                faceRetLab = np.where(finalPoseList[-1]==False, faceRetLab, 'Invalid Pose')
                faceRetLab = np.where(probBArea >= img.smallFaceAreaPercentage, faceRetLab, 'Small Face')
                finalFaceProability.append(faceMaxProb.tolist())
                finalFaceLabels.append(faceRetLab.tolist())
        return finalBBOXsRaw, finalRawLandmarks, finalFaceProability, finalFaceLabels, finalSpoofList
        
# Stream Handler
class streamClient:
    def __init__(self, clientId: string) -> None:
        """
        This method is used to handle stream client for trackers

        Arguments:
        ==========
        cliendId : A consistent randomly generated client ID
        """
        self.clientId = clientId
        self.trackers = dict()
        
    def __call__(self, inferenceObj: Inference, imgList: list, detectionConfidence: list, nmsThreshold: list, boxCushion: list, boxArea: list, recogThres: list, fakeThres: list, trackerFrames: list, trackerIds: list) -> dict:
        """
        This method is used to run the inference on the given stream

        Arguments:
        ==========
        inferenceObj : Inference model object to get raw faces inference
        imgList : Image object pool, specifically designed for model inference
        detectionConfidence : Bounding box confidence threshold
        nmsThreshold : Non-Max-Suppression threshold in bounding box detection
        boxCushion : Bounding box padding pixels, which are added to bounding box dimensions
        boxArea : Percentage area of bounding box, lesser than which face is considered as small
        recogThres : Face recognition threshold
        fakeThres : Face fakeness threshold
        trackerFrames : Number of frames of the stream to track
        trackerIds : Unique consistent random tracker Ids to track frames in a stream
        """
        for img, dC, nmsT, bC, bA, rT, fT, tF, tI in zip(imgList, detectionConfidence, nmsThreshold, boxCushion, boxArea, recogThres, fakeThres, trackerFrames, trackerIds):
            for i in list(self.trackers.keys()):
                if i not in trackerIds:
                    del self.trackers[i]
                    torch.cuda.empty_cache()
            if tI in list(self.trackers.keys()):
                self.trackers[tI]['counter'] += 1
                self.trackers[tI]['toInfer'] = False
                if (self.trackers[tI]['counter'] > tF) or (self.trackers[tI]['f2T']!=tF):
                    self.trackers[tI] = {
                        'counter': 0,
                        'toInfer': True,
                        'f2T': tF,
                        'image': imgData(img, tI, dC, nmsT, bC, bA, rT, fT, tF),
                        'currentLandmarks': list(),
                        'currentBBox': list(),
                        'currentBoxProbability': list(),
                        'currentFaceProbability': list(),
                        'currentFaceLabel': list(),
                        'currentSpoofs': list(),
                        'trackerPool': list()
                    }
                else:
                    self.trackers[tI]['image'] = imgData(img, tI, dC, nmsT, bC, bA, rT, fT, tF)
            else:
                self.trackers[tI] = {
                    'counter': 0,
                    'toInfer': True,
                    'f2T': tF,
                    'image': imgData(img, tI, dC, nmsT, bC, bA, rT, fT, tF),
                    'currentLandmarks': list(),
                    'currentBBox': list(),
                    'currentBoxProbability': list(),
                    'currentFaceProbability': list(),
                    'currentFaceLabel': list(),
                    'currentSpoofs': list(),
                    'trackerPool': list()
                }
        toInferImgs = dict()
        for i in list(self.trackers.keys()):
            if self.trackers[i]['toInfer'] == True:
                toInferImgs[i] = self.trackers[i]['image']
            else:
                for p in range(len(self.trackers[i]['trackerPool'])):
                    self.trackers[i]['trackerPool'][p].update(self.trackers[i]['image'].rawData)
                    newRect = self.trackers[i]['trackerPool'][p].get_position()
                    self.trackers[i]['currentBBox'][p][0] = newRect.left()
                    self.trackers[i]['currentBBox'][p][1] = newRect.top()
                    self.trackers[i]['currentBBox'][p][2] = newRect.right()
                    self.trackers[i]['currentBBox'][p][3] = newRect.bottom()
        if len(toInferImgs)>0:
            finalBBOXsRaw, finalRawLandmarks, finalFaceProability, finalFaceLabels, finalSpoofList = inferenceObj(list(toInferImgs.values()))
            for i, j, k, keyId, finalLmks, finalSpoof in zip(finalBBOXsRaw, finalFaceProability, finalFaceLabels, list(toInferImgs.keys()), finalRawLandmarks, finalSpoofList):
                self.trackers[keyId]['currentBoxProbability'] = [fp[0] for fp in i]
                self.trackers[keyId]['currentBBox'] = [fb[1:] for fb in i]
                self.trackers[keyId]['currentFaceProbability'] = j
                self.trackers[keyId]['currentFaceLabel'] = k
                self.trackers[keyId]['currentSpoofs'] = finalSpoof
                self.trackers[keyId]['currentLandmarks'] = finalLmks
                for t in self.trackers[keyId]['currentBBox']:
                    ttracker = dlib.correlation_tracker()
                    ttracker.start_track(self.trackers[keyId]['image'].rawData, dlib.rectangle(int(t[0]), int(t[1]), int(t[2]), int(t[3])))
                    self.trackers[keyId]['trackerPool'].append(ttracker)
        outJson = dict()
        for i in trackerIds:
            outBBox = list()
            outBBoxProbability = list()
            outFaces = list()
            outSpoofs = list()
            outFacesProbability = list()
            tbox = list()
            for cbox in self.trackers[i]['currentBBox']:
                cbox[0] /= self.trackers[i]['image'].dataShape[1]
                cbox[1] /= self.trackers[i]['image'].dataShape[0]
                cbox[2] /= self.trackers[i]['image'].dataShape[1]
                cbox[3] /= self.trackers[i]['image'].dataShape[0]
                tbox.append(cbox)
            outBBox.append(tbox)
            outBBoxProbability.append(self.trackers[i]['currentBoxProbability'])
            outFaces.append(self.trackers[i]['currentFaceLabel'])
            outSpoofs.append(self.trackers[i]['currentSpoofs'])
            outFacesProbability.append(self.trackers[i]['currentFaceProbability'])
            outJson[i] = {
                'bbox': outBBox,
                'boxProbability': outBBoxProbability,
                'faces': outFaces,
                'facesProbability': outFacesProbability,
                'spoofs': outSpoofs
            }
        return outJson

if __name__=="__main__":
    print('FACE DETECTION, RECOGNTION & TRACKING MODULE')
