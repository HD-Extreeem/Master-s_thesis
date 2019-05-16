# ----------------------------------------------------|
# TO RUN TYPE: python all_in_one.py --image park1.png-|
# ----------------------------------------------------|


import cv2
import argparse
import numpy as np
import time
import os
import operator
#import caffe
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to input image')
args = ap.parse_args()

classes = None

mob_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

alex_classes = ["free", "busy"]

with open("../../yolo3/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

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

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 4)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

def yolo3_classify(image_yolo3, classes, COLORS):
    Width = image_yolo3.shape[1]
    Height = image_yolo3.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("../../yolo3/yolov3.cfg", "../../yolo3/yolov3.weights")

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
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        print("{}: {:.2f}%".format(str(classes[class_ids[i]]), confidences[i]*100))
        #draw_prediction(image_yolo3, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        cv2.rectangle(image_yolo3,(x-10,y-10),(x+w,y+h),(0,255,0),10)
        #cv2.putText(image_yolo3, str(classes[class_ids[i]]), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        #cv2.putText(image_yolo3, "%" +str(int(confidences[i]*100)), (x+550,y-10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 6)


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
    blob = cv2.dnn.blobFromImage(cv2.resize(image_ssd_mobilenet, (500, 500)), 0.007843, (500, 500), (127.5, 127.5, 127.5))
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    print("[INFO] Caffe MobileNetSSD took {:.6f} seconds".format(end - start))
    print(detections)
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
        
        # draw the prediction on the frame
        label = "{}: {:.2f}%".format(mob_classes[idx],confidence * 100)
        cv2.rectangle(image_ssd_mobilenet, (startX, startY), (endX, endY), COLORS[idx], 4)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image_ssd_mobilenet, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 4)
        print("{}: {:.2f}%".format(mob_classes[idx],confidence * 100))

# |--------------------------------------|
# |-----Functions for AlexNet-Caffe------|
# |--------------------------------------|
def alexnet_classify(image_alexnet, alex_classes, COLORS):

    #net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt", "alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_870.caffemodel")
    
    net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark-Trained-Models/mAlexNet-on-CNRPark/deploy.prototxt", "alexnet/CNRPark-Trained-Models/mAlexNet-on-CNRPark/snapshot_iter_942.caffemodel")

    #net = cv2.dnn.readNetFromCaffe("alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/deploy.prototxt", "alexnet/CNRPark+EXT_Trained_Models_mAlexNet-2/mAlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val/snapshot_iter_870.caffemodel")
    
    (fH, fW) = image_alexnet.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_alexnet, (227, 227)), 1/255, (227, 227), (104, 117, 123))
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
    idx = int(detections[0, 0])
    print(idx)
    for (i, idx) in enumerate(idxs):
        # draw the top prediction on the input image
        print(idx)
        if i == 0:
            text = "Label: {}, {:.2f}%".format(alex_classes[idx], detections[0][idx] * 100)
            cv2.putText(image_alexnet, text, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

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



image_yolo3 = cv2.imread(args.image)
#image_yolo3_tiny = cv2.imread(args.image)
#image_ssd_mobilenet = cv2.imread(args.image)
#image_alexnet = cv2.imread(args.image)

yolo3_classify(image_yolo3, classes, COLORS)
#tiny_yolo3_classify(image_yolo3_tiny, classes, COLORS)
##mobilenet_classify(image_ssd_mobilenet, mob_classes, COLORS)
#alexnet_classify(image_alexnet, alex_classes, COLORS)
image_yolo3 = cv2.resize(image_yolo3, (800, 600))
cv2.imshow("YOLO3", image_yolo3)
#cv2.imshow("TINY-YOLO3", image_yolo3_tiny)
#cv2.imshow("SSD-MOBILENET", image_ssd_mobilenet)
#cv2.imshow("AlexNet", image_alexnet)
cv2.waitKey()
    
cv2.imwrite("YOLO3.jpg", image_yolo3)
#cv2.imwrite("TINY-YOLO3.jpg", image_yolo3_tiny)
#cv2.imwrite("SSD-MOBILENET.jpg", image_ssd_mobilenet)
#cv2.imwrite("AlexNet", image_alexnet)

cv2.destroyAllWindows()
