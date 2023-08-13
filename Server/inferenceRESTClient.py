#!/usr/bin/env python3

"""
MODEL INFERENCE CLIENT (REST API)
=================================

Following program is used to perform inference on subject data using REST API
"""


# %%
# Importing Libraries
import argparse
import json
import string
import base64
from io import BytesIO
from PIL import Image
import requests

# %%
# Inference Tools
class Detector:
    def __init__(self, url: string, clientID: string) -> None:
        """
        This class is used to initialize classification client
        
        Arguments
        =========
        url : Inference Server URL
        clientID : Client ID for Inference Tracking
        """
        self.server_url = url
        self.clientID = clientID
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def __encodeImage__(self, addr: string) -> string:
        """
        This method is used to encode image for inference
        
        Arguments
        =========
        addr : Absolute address of image
        
        Outputs
        =======
        Encoded Image
        """
        img = Image.open(addr)
        byt1 = BytesIO()
        img.save(byt1, format = 'JPEG')
        bdat = byt1.getvalue()
        return base64.b64encode(bdat).decode('utf-8')
    
    def __call__(self, imgLinks: list, detectionConfidence: list, nmsThreshold: list, boxCushion: list, smallFace: list, recogThres: list, spoofThres: list, trackerFrames: list) -> tuple:
        """
        This method is used to handle image's inference from inference server
        
        Arguments
        =========
        imgLinks : Absolute address of image's in the form on single string or list
        
        Outputs
        =======
        Infered data & response status code
        """
        imgStack = list()
        for i in imgLinks:
            imgStack.append(self.__encodeImage__(i))
        resp = requests.post(self.server_url, headers = self.headers, json = {
            'clientID': self.clientID,
            'images': imgStack,
            'detectionThreshold': detectionConfidence,
            'nmsThreshold': nmsThreshold,
            'boxCushion': boxCushion,
            'smallFacePercentage': smallFace,
            'recognitionThreshold': recogThres,
            'fakenessThreshold': spoofThres,
            'trackingFrames': trackerFrames
        })
        return resp.json(), resp.status_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Face Detection & Recognition Inference Client.')
    parser.add_argument('-l', '--link', type = str, help = 'Absolute Address of Subject Images', required = True)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to REST Server => IP:Port/route', required = True)
    parser.add_argument('-det', '--detectionThreshold', type = float, help = 'Face Detection Confidence Threshold', default = 0.6)
    parser.add_argument('-rec', '--recognitionThreshold', type = float, help = 'Face Recognition Confidence Threshold', default = 0.5)
    parser.add_argument('-spof', '--spoofingThreshold', type = float, help = 'Face Anti-Spoofing Confidence Threshold', default = 0.5)
    parser.add_argument('-nms', '--nmsThreshold', type = float, help = 'Face Detection Non-Max Suppression', default = 0.5)
    parser.add_argument('-bcus', '--boxCushion', type = int, help = 'Face Detection Box-Cushion', default = 0)
    parser.add_argument('-sFP', '--smallFacePercent', type = float, help = 'Small Face Percentage', default = 1.0)
    parser.add_argument('-trF', '--trackerFrames', type = int, help = 'Face Detection Tracker Frames', default = 0)
    args = vars(parser.parse_args())
    det = Detector(args['server_ip'], 'REST-API-TEST')
    res, scode = det([args['link']], [args['detectionThreshold']], [args['nmsThreshold']], [args['boxCushion']], [args['smallFacePercent']], [args['recognitionThreshold']], [args['spoofingThreshold']], [args['trackerFrames']])
    print(res)