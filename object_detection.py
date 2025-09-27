import cv2
import sys
import os
import matplotlib
import numpy as np
from collections import Counter

############################################ Setup YOLO v3 ######################################################
lbl_file        = 'models/yolov3.txt'
classes         = open(lbl_file).read().strip().split("\n")

yoloconfig      = 'models/yolov3.cfg'
yoloweights     = 'models/yolov3.weights'
net             = cv2.dnn.readNet(yoloweights,yoloconfig)

############################################# YOLO Detection #####################################################

def yoloV3Detect(img, scFactor=1/255, nrMean=(0,0,0), RBSwap=True, scoreThres=0.5, nmsThres=0.4):
    
    # Input validation
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        print("Invalid input image for YOLO detection")
        return [], []
    
    try:
        ########################## Create blob #########################
        blob = cv2.dnn.blobFromImage(image=img, 
                                    scalefactor=scFactor, 
                                    size=(416, 416), 
                                    mean=nrMean, 
                                    swapRB=RBSwap, 
                                    crop=False)
        
        ########################## Prediction ############################
        def getOutputLayers(net): 
            layers = net.getLayerNames() 
            outLayers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()] 
            return outLayers

        net.setInput(blob) 
        outLyrs = getOutputLayers(net) 
        preds = net.forward(outLyrs)

        ############### Extract information from the output ###############
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        classId = [] 
        confidences = [] 
        boxes = []

        for scale in preds: 
            for pred in scale: 
                scores = pred[5:] 
                clss = np.argmax(scores) 
                confidence = scores[clss]

                if confidence > scoreThres: 
                    xc = int(pred[0]*imgWidth) 
                    yc = int(pred[1]*imgHeight) 
                    w = int(pred[2]*imgWidth) 
                    h = int(pred[3]*imgHeight) 
                    x = xc - w/2
                    y = yc - h/2
                    
                    classId.append(clss) 
                    confidences.append(float(confidence)) 
                    boxes.append([x, y, w, h])
        
        # Check if any detections were made
        if len(boxes) == 0:
            return [], []
        
        ############### Non-maximal suppresion (NMS) #####################
        selected = cv2.dnn.NMSBoxes(bboxes=boxes, 
                                    scores=confidences, 
                                    score_threshold=scoreThres, 
                                    nms_threshold=nmsThres)
        
        # FIXED: Check if selected is not empty
        if len(selected) == 0:
            return [], []
        
        # FIXED: Handle the case where selected might be a different shape
        if isinstance(selected, tuple):
            selected = selected[0] if len(selected) > 0 else []
        
        if len(selected) == 0:
            return [], []
        
        # FIXED: Proper indexing for selected boxes
        try:
            if selected.ndim == 2:
                fboxes = [boxes[j] for j in selected[:, 0]]
                fclasses = [str(classes[classId[j]]) for j in selected[:, 0]]
            else:
                fboxes = [boxes[j] for j in selected]
                fclasses = [str(classes[classId[j]]) for j in selected]
        except (IndexError, TypeError) as e:
            print(f"Error in NMS indexing: {e}")
            return [], []
        
        return [fboxes, fclasses]
        
    except Exception as e:
        print(f"Error in YOLO detection: {e}")
        return [], []