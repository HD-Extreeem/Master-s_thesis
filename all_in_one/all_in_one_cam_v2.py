# -----------------------------------------------------------------|
#   TO RUN TYPE:                                                   |
#   python all_in_one_cam_v2.py --image park4.png --url yes            |
# -----------------------------------------------------------------|

import cv2
import argparse
import numpy as np
import time
import os
import operator
import caffe_classes_alex
from urllib.request import urlopen
from copy import copy
import requests
from PIL import Image
import io
import json

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
ap.add_argument('-u', '--url', required=True,
                    help='Use URL image')
args = ap.parse_args()

classes = None

url_img = 'http://root:ateapass@192.168.0.90/axis-cgi/jpg/image.cgi?resolution=1920x1080'

interval = 30

mob_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

alex_classes = ["free", "busy"]
#alex_classes = caffe_classes_alex.class_names

with open("yolo3/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_img(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def get_cam_img():
    r = requests.get(url_img)
    print("Status code"+str(r.status_code))
    if r.status_code == 200:
        image = np.asarray(bytearray(r.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image

def read_put_coord(startX,endX,startY,endY):
    put = "StartX-Endx: {} - {} ; StartY-EndY: {} - {} \n".format(startX , endX, startY, endY)
    with open("coord.txt") as f :
        content = f.read()
        if(put in content):
            print("EXISTS ALREADY")
        else:
            f.close()
            fp = open("coord.txt","a")
            fp.write(put)
            fp.close()

def put_coord(startX,endX,startY,endY):
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    already = False
    for element in data["parking"]:
        
        if(element["startX"] == startX):
            already = True
            element["time"] += 30
            break

    if not already:
        info = {
            "startX": startX,
            "endX": endX,
            "startY": startY,
            "endY": endY,
            "time": 0
        }
        data["parking"].append(info)

    with open('coordinates.json', 'w') as file:
        json.dump(data, file)



# |-------------------------------------------|
# |-----Functions for YOLO3 & Tiny YOLO3------|
# |-------------------------------------------|

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
 
    label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
    #label = "{}: {:.2f}%".format(mob_classes[idx],confidence * 100)
    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def yolo3_classify(image_yolo3, classes, COLORS):
    Width = image_yolo3.shape[1]
    Height = image_yolo3.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("yolo3/yolov3.cfg", "yolo3/yolov3.weights")

    blob = cv2.dnn.blobFromImage(image_yolo3, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    start = time.time()
    outs = net.forward(get_output_layers(net))
    end = time.time()
    print("[INFO] YOLO3 took {:.6f} seconds".format(end - start))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    #label = "{}: {:.2f}%".format(class_ids[i], confidence * 100)


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print("X: {}, Y: {}, X+W: {}, Y+H: {}".format( x, y, x+w, y+h))
        print("{}: {:.2f}%".format(str(classes[class_ids[i]]), confidences[i]*100))
        draw_prediction(image_yolo3, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        put_coord(x, y, x+w, y+h)

#TINY-YOLO3
def tiny_yolo3_classify(image_yolo3_tiny, classes, COLORS):
    Width = image_yolo3_tiny.shape[1]
    Height = image_yolo3_tiny.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("yolo3/yolov3-tiny.cfg", "yolo3/yolov3-tiny.weights")

    blob = cv2.dnn.blobFromImage(image_yolo3_tiny, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    start = time.time()
    outs = net.forward(get_output_layers(net))
    end = time.time()
    print("[INFO] TINY-YOLO3 took {:.6f} seconds".format(end - start))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                #print("X: {}, Y: {}, W: {}, H: {}".format( x, y, w, h))
                #print("{}: {:.2f}%".format(str(class_id), confidence*100))
                #label = "{}: {:.2f}%".format(class_ids[i], confidence * 100)


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print("{}: {:.2f}%".format(str(classes[class_ids[i]]), confidences[i]*100))
        draw_prediction(image_yolo3_tiny, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# |--------------------------------------|
# |-----Functions for SSD-MobileNet------|
# |--------------------------------------|
def mobilenet_classify(image_ssd_mobilenet, mob_classes, COLORS):
    net = cv2.dnn.readNetFromCaffe("mobilenet/MobileNetSSD_deploy.prototxt.txt", "mobilenet/MobileNetSSD_deploy.caffemodel")
    (fH, fW) = image_ssd_mobilenet.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_ssd_mobilenet, (400,400)), 0.007843, (300,300), (127.5, 127.5, 127.5))
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    print("[INFO] Caffe MobileNetSSD took {:.6f} seconds".format(end - start))
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence`
        # is greater than the minimum confidence
        if confidence < 0.4:
            continue
                
        # otherwise, extract the index of the class label from
        # the `detections`, then compute the (x, y)-coordinates
        # of the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        
        dims = np.array([fW, fH, fW, fH])
        box = detections[0, 0, i, 3:7] * dims
        (startX, startY, endX, endY) = box.astype("int")
        print("StartX: {}, EndX: {}, StartY: {}, EndY: {}".format(startX, startY, endX, endY))
        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(mob_classes[idx],confidence * 100)
        cv2.rectangle(image_ssd_mobilenet, (startX, startY), (endX, endY), COLORS[idx], 1)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image_ssd_mobilenet, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[idx], 1)
        print("{}: {:.2f}%".format(mob_classes[idx],confidence * 100))


# |--------------------------------------|
# |-----Functions for AlexNet-Caffe------|
# |--------------------------------------|
def alexnet_classify(image_alexnet, alex_classes, COLORS):

    #net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt", "alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_870.caffemodel")
    
    #net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark-Trained-Models/mAlexNet-on-CNRPark/deploy.prototxt", "alexnet/CNRPark-Trained-Models/mAlexNet-on-CNRPark/snapshot_iter_942.caffemodel")

    net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt", "alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_870.caffemodel")
    #net = cv2.dnn.readNet("alexnet/CNRPark+EXT_Trained_Models_AlexNet/AlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt", "alexnet/CNRPark+EXT_Trained_Models_AlexNet/AlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_990.caffemodel")
    #net = cv2.dnn.readNet("alexnet/alexnet.cfg", "alexnet/alexnet.weights")

    (fH, fW) = image_alexnet.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_alexnet, (227, 227)), 1. / 256, (227, 227), (104, 117, 123))
    #blob = cv2.dnn.blobFromImage(cv2.resize(image_alexnet, (224, 224)), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    
    #index, value = max(enumerate(detections[0]), key=operator.itemgetter(1))
    
    #print (index) # prints the index of max probability
    #print (value) # prints the max probability
    
    print("[INFO] Caffe AlexNet took {:.6f} seconds".format(end - start))
    print (detections)
    idxs = np.argsort(detections[0])[::-1][:5]
    
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(alex_classes[idx], detections[0][idx] * 100)
            cv2.putText(image_alexnet, text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, alex_classes[idx], detections[0][idx]))
        # display the predicted label + associated probability to the
        # console

'''
    for i in np.arange(0, detections.shape[2]):
        print(i)
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence`
        # is greater than the minimum confidence
        if confidence < 0.2:
            continue
        
        # otherwise, extract the index of the class label from
        # the `detections`, then compute the (x, y)-coordinates
        # of the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        dims = np.array([fW, fH, fW, fH])
        box = detections[0, 0, i, 3:7] * dims
        (startX, startY, endX, endY) = box.astype("int")
        
        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(alex_classes[idx],confidence * 100)
        cv2.rectangle(image_alexnet, (startX, startY), (endX, endY), COLORS[idx], 4)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image_alexnet, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 4)
        print("{}: {:.2f}%".format(alex_classes[idx],confidence * 100))
'''

#----------------------------------------------
# Check what mode is chosen,URL or local image-
#----------------------------------------------

while(True):
    if (args.url == "yes"):
        start = time.time()
        #img_url = "https://www.umass.edu/transportation/sites/default/files/IMG_2943_0.JPG"
        img_url = "https://i.dailymail.co.uk/i/pix/2016/03/28/11/329BDAA800000578-3512279-image-a-36_1459160823373.jpg"
        #start = time.time();
        #image = get_cam_img()
        image = get_img(img_url)
        #end = time.time();
        #print("[INFO] Download took {:.6f} seconds".format(end - start))
        
        (image_yolo3, image_yolo3_tiny, image_ssd_mobilenet) = (copy(image),copy(image),copy(image))
        #image_alexnet = image

#-------------------------------------
# Methods for classifying the objects-
#-------------------------------------
        yolo3_classify(image_yolo3, classes, COLORS)
        tiny_yolo3_classify(image_yolo3_tiny, classes, COLORS)
        mobilenet_classify(image_ssd_mobilenet, mob_classes, COLORS)
        
        #alexnet_classify(image_alexnet, alex_classes, COLORS)

#-----------------
# Show the images-
#-----------------

        cv2.imshow("YOLO3", image_yolo3)
        cv2.imshow("TINY-YOLO3", image_yolo3_tiny)
        cv2.imshow("SSD-MOBILENET", image_ssd_mobilenet)
        #cv2.imshow("AlexNet", image_alexnet)
        
    while((time.time()-start < interval) and (time.time()-start>0)):
        time.sleep(1)
        print(time.time()-start)
