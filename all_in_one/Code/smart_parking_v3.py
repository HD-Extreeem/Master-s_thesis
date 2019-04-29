# -----------------------------------------------------------------|
#   TO RUN TYPE:                                                   |
#   python smart_parking_v2.py
# -----------------------------------------------------------------|
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os
import operator
from urllib.request import urlopen
from copy import copy
import requests
from PIL import Image
import io
import json
import serial
import math

classes = None
#ser = serial.Serial('/dev/ttyACM1',9600,5)
url_img = 'http://root:ateapass@192.168.0.90/axis-cgi/jpg/image.cgi?resolution=1920x1080'
ID = 1
interval = 5
free_spaces = -1
vehicle_boxes=[]
park_boxes=[]
classify_treshold = 0.5
imgpath1 = "../Image/Test1.jpg"
image1 = cv2.imread(imgpath1)
image1 = cv2.resize(image1, (1920, 1080))
imgpath2 = "../Image/Test2.jpg"
image2 = cv2.imread(imgpath2)
image2 = cv2.resize(image2, (1920, 1080))
imgpath3 = "../Image/3.png"
image3 = cv2.imread(imgpath3)
imgpath4 = "../Image/4.png"
image4 = cv2.imread(imgpath4)
imgpath5 = "../Image/5.png"
image5 = cv2.imread(imgpath5)
imgpath6 = "../Image/park_stor.jpg"
image6 = cv2.imread(imgpath6)
imgpath7 = "../Image/park_cnr.png"
image7 = cv2.imread(imgpath7)

with open("../yolo3/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_cam_img():
    r = requests.get(url_img)
    print("Status code"+str(r.status_code))
    if r.status_code == 200:
        image = np.asarray(bytearray(r.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image

def put_coord(startX,endX,startY,endY,IoU,ID,state):
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    info = {
        "startX": startX,
        "endX": endX,
        "startY": startY,
        "endY": endY,
        "IoU": IoU,
        "ID":ID,
        "State":state
    }
    data["parking"].append(info)
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)

def put_boxes (boxes):
    boxes_new = calculate_park_IoU(boxes)
    boxes_new = sort_parking_id(boxes_new)
    with open('coordinates.json', 'w') as file:
        data = {"parking":[]}
        json.dump(data, file)
    for i in boxes_new:
        if len(i)==6:
            i.append("Busy")
            put_coord( i[0], i[1], i[2], i[3], i[4], i[5], i[6])
        else:
            put_coord( i[0], i[1], i[2], i[3], i[4], i[5], i[6])

def sort_parking_id(boxes):
    
    sorted_boxes = sorted(boxes,key=lambda x : (x[0],x[2]))
    num = 1
    for i in sorted_boxes: 
        #print(type(boxes[0]))
        #label = "{:.2f}".format(num)
        length = len(i)
        if length==5:
            i.append(num)
        else:
            i[5]=num
        num+=1
    return sorted_boxes
    


def load_coord():
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    slot_box =[]
    for element in data["parking"]:
        slot_box.append([element["startX"],element["endX"],element["startY"],element["endY"],element["IoU"],element["ID"],element["State"]])
    return slot_box

def replace_coord(old_startX,old_startY,startX,endX,startY,endY):
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    for element in data["parking"]:
        if(element["startX"] == old_startX) and ((element["startY"] == old_startY)):
            element["startX"] = startX
            element["endX"] = endX
            element["startY"] = startY
            element["endY"] = endY
            break
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)

def calculate_IoU(boxA,boxB):
    XA= max(boxA[0],boxB[0])
    XB= min(boxA[1],boxB[1])
    YA= max(boxA[2],boxB[2])
    YB= min(boxA[3],boxB[3])

    #calculate the intern area
    interArea = max(0,XB-XA)* max(0,YB-YA)

    #calculate the area of both boxes
    boxAArea = (boxA[1]-boxA[0])*(boxA[3]-boxA[2])
    boxBArea = (boxB[1]-boxB[0])*(boxB[3]-boxB[2])

    #calculate the intersection-over union area
    iou= interArea / float(boxAArea+boxBArea-interArea)
    #print(iou)
    return iou

def calculate_park_IoU(places):
    for i in places:
        IoU_max=0
        for j in places:
            IoU = calculate_IoU(i,j)
            if IoU_max < IoU and IoU < 0.99:
                IoU_max=IoU
        if(len(i)==4):
            i.append(IoU_max)
        else:
            i[4]=IoU_max
    return places

def calculate_free_spots(image,p_boxes,v_boxes):
    free_space=0
    for slot in p_boxes:
        IoU_max = 0
        for vehicle in v_boxes:
            IoU = calculate_IoU(slot,vehicle)
            if IoU_max < IoU:
               IoU_max=IoU
        x1 =int(slot[0])
        x2 =int(slot[1])
        y1 =int(slot[2])
        y2 =int(slot[3])
        Id = slot[5]
        State =""
        print("P place coordinates are :",slot)
        print("Maximum IoU is: ",IoU_max)
        print("Parking space IoU is: ",slot[4])
        if IoU_max < slot[4]+0.10:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),10)
           free_space += 1
           State="Free"
        if IoU_max > 0.70:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),10)
           State = "Busy"
        if len(slot)<7:
            slot.append(State)
        else:
            slot[6]=State
        cv2.putText(image, str(Id), (x1+50,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
    put_boxes(p_boxes)
    print("Number of free parking space is: ",free_space)
    print("Number of total parking space is: ",len(p_boxes))
    return free_space


def compare_spots(p_boxes,v_boxes):
    print("Number of p boxes is:", len(p_boxes))
    print("Number of v boxes is:", len(v_boxes))
    print(p_boxes)
    for vehicle in v_boxes:
        IoU_max = 0
        for park in p_boxes:
            IoU = calculate_IoU(park,vehicle)
            if IoU_max < IoU:
               IoU_max=IoU
               print(park)
               x1,x2,y1,y2,temp_iou,Id,state = park
        X1,X2,Y1,Y2,v_type = vehicle
        if IoU_max < (temp_iou + 0.20):
            print("Temp IoU = ",temp_iou)
            print("Lägger till en ny p_plats")
            p_boxes.append([X1,X2,Y1,Y2])
            put_boxes(p_boxes)
            print("New park space:", X1,X2,Y1,Y2)
            print("IoU is: ", IoU_max )
            p_boxes.clear()
            p_boxes = load_coord()
        elif IoU_max > 0.80 and v_type != "motorcycle" and v_type != "bus":
            print("Tar medelvärdet och uppdaterar, IoU is:", IoU_max)
            X1=(x1+X1)/2
            X2=(x2+X2)/2
            Y1=(y1+Y1)/2
            Y2=(y2+Y2)/2
            replace_coord(x1,y1,X1,X2,Y1,Y2)
            p_boxes.clear()
            p_boxes = load_coord()
    print("number OF TOTAL P PLACE : ", len(p_boxes))
    return p_boxes

def compare_boxes(BOXA,BOXB):
    #Xa1 = BOXA[0]
    #Xa2 = BOXA[1]
    #Ya1 = BOXA[2]
    #Ya2 = BOXA[3]
    #Xb1 = BOXB[0]
    #Xb2 = BOXB[1]
    #Yb1 = BOXB[2]
    #Yb2 = BOXB[3]
    #dist1 = math.sqrt( (Xa1 - Xb1)**2 + (Ya1 - Yb1)**2 )
    #dist2 = math.sqrt( (Xa2 - Xb2)**2 + (Ya2 - Yb2)**2 )
    #distRef = math.sqrt( (Xa1 - Xa2)**2 + (Ya1 - Ya2)**2 )
    #if (dist1 < (distRef/20)) and (dist2 < (distRef/20)):
    if calculate_IoU(BOXA,BOXB)>0.9:
        return True
    else:
        return False

def compare_list (listA,listB):
    lengthA = len(listA)
    lengthB = len(listB)
    check =0
    for a in listA:
        for b in listB:
            if compare_boxes(a,b):
                check += 1
                break
    if (lengthA == check) and (lengthB == lengthA):
        return True
    else:
        return False


# |-------------------------------------------|
# |-----Functions for YOLO3------|
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

def yolo3_classify(image_yolo, classes, COLORS):
    Width = image_yolo.shape[1]
    Height = image_yolo.shape[0]
    scale = 0.00392
    net = cv2.dnn.readNet("../yolo3/yolov3.cfg", "../yolo3/yolov3.weights")
    blob = cv2.dnn.blobFromImage(image_yolo, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(get_output_layers(net))
    end = time.time()
    print("[INFO] YOLO3 took {:.6f} seconds".format(end - start))
    class_ids = []
    confidences = []
    boxes = []
    BOX =[]
    nms_threshold = 0.2

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > classify_treshold:
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, classify_treshold, nms_threshold)

    for i in indices:
        
        i = i[0]
        if(str(classes[class_ids[i]]) == "car" or str(classes[class_ids[i]]) == "truck" or str(classes[class_ids[i]]) == "motorcycle" or str(classes[class_ids[i]]) == "bus"):
            box = boxes[i]
            x1 = box[0]
            x2 = box[0]+box[2]
            y1 = box[1]
            y2 = box[3]+box[1]
            #print("Classes = ",str(classes[class_ids[i]]))
            BOX.append([x1,x2,y1,y2,str(classes[class_ids[i]])])
        #print("X: {}, Y: {}, X+W: {}, Y+H: {}".format( x1, y1, x2, y2))
        #print("{}: {:.2f}%".format(str(classes[class_ids[i]]), confidences[i]*100))
        #draw_prediction(image_yolo, class_ids[i], confidences[i], round(x1), round(y1), round(x2), round(y2))

    #image_yolo = cv2.resize(image_yolo, (800, 600))
    #cv2.imshow("YOLO3", image_yolo)
    #cv2.waitKey()
    return BOX



#----------------------------------------------
# Check what mode is chosen,URL or local image-
#----------------------------------------------
with open('coordinates.json', 'w') as file:
        data = {"parking":[]}
        json.dump(data, file)

print("Program starts")

while(True):
    try:
        x = int(input("Enter the image number...")) #integer input
    except ValueError:
        x=1
        
    if x==1:
        image_yolo3 = copy(image1)
        img = copy(image1)
    elif x==2:
        image_yolo3 = copy(image2)
        img = copy(image2)
    elif x==3:
        image_yolo3 = copy(image3)
        img = copy(image3)
    elif x==4:
        image_yolo3 = copy(image4)
        img = copy(image4)
    elif x==5:
        image_yolo3 = copy(image5)
        img = copy(image5)
    elif x==6:
        image_yolo3 = copy(image6)
        img = copy(image6)
    elif x==7:
        image_yolo3 = copy(image7)
        img = copy(image7)
    start = time.time()
    print("Loads coordinates")
    park_boxes = load_coord()
    #print("Json coordinats in the beginning", park_boxes)

    #image = get_cam_img()        

#-------------------------------------
# Methods for classifying the objects-
#-------------------------------------
    print("Classifying with Yolov3")
    vehicle_boxes = yolo3_classify(img, classes, COLORS)
    if len(vehicle_boxes) > 0:
        print("Number of detected vehicle is: ",len(vehicle_boxes))
        while True:
            time.sleep(interval)
            print("Second round classifying just to be sure")
            vehicle_boxes2 = yolo3_classify(img, classes, COLORS)
            if(compare_list(vehicle_boxes,vehicle_boxes2)):
                print("We are sure that at least one vehicle is there")
                break
            else: 
                print("We will classify again")
                vehicle_boxes = vehicle_boxes2
        if len(park_boxes) == 0:
            print("This is new program put every box as a parking space")
            park_boxes = copy(vehicle_boxes)
            put_boxes(park_boxes)
        else:
            print("Comparing the spots")
            park_boxes = load_coord()
            print(park_boxes)
            park_boxes = compare_spots(park_boxes,vehicle_boxes)
    #print("Json coordinats in the end", park_boxes)
    print("Calculating free spots")
    new_free_spaces = calculate_free_spots(image_yolo3,park_boxes,vehicle_boxes)
    

#-----------------
# Show the images-
#-----------------

    #cv2.imshow("YOLO3", image_yolo3)
    #cv2.waitKey()
    while((time.time()-start < interval) and (time.time()-start>0)):
        time.sleep(1)
        #print(time.time()-start)
   # if index == 7:
    #    image_yolo3 = cv2.resize(image_yolo3, (800, 600))
     #   cv2.imshow("YOLO3", image_yolo3)
      #  cv2.waitKey()
    #index += 1
    image_yolo3 = cv2.resize(image_yolo3, (800, 600))
    cv2.imshow("Parking Spaces", image_yolo3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vehicle_boxes.clear()
    print(park_boxes)
    if(free_spaces != new_free_spaces):
       lora_message = "New-"+str(ID)+"-"+str(len(park_boxes))+"-"+str(new_free_spaces)
       print(lora_message)
       #ser.write(lora_message.encode())
       #str1=ser.readline()
       #print(str1)
       free_spaces = new_free_spaces

