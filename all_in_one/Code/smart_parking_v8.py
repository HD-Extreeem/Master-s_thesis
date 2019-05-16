# -----------------------------------------------------------------|
#   TO RUN TYPE:                                                   |
#   python smart_parking_v5.py
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
from scipy.spatial import distance
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

classes = None
#ser = serial.Serial('/dev/badgerboard',9600,5)
url_img = 'http://root:ateapass@192.168.0.90/axis-cgi/jpg/image.cgi?resolution=1920x1080'
ID = 1
interval = 20
dbscan_ready=True
min_distance = 10000
nSamples = 0
free_spaces = -1
vehicle_boxes=[]
park_boxes=[]
classify_treshold = 0.3
time_thresh = 1200 # = 20 min

image1 = cv2.imread("../Image/image1.jpg")
image2 = cv2.imread("../Image/image2.jpg")
image3 = cv2.imread("../Image/image3.jpg")
image4 = cv2.imread("../Image/image4.jpg")
image5 = cv2.imread("../Image/image5.jpg")
image6 = cv2.imread("../Image/image6.jpg")
image7 = cv2.imread("../Image/image7.jpg")
image8 = cv2.imread("../Image/image8.jpg")
image9 = cv2.imread("../Image/image9.jpg")
#image6 = cv2.imread("../Image/park_stor.jpg")
#image7 = cv2.imread("../Image/park_cnr.png")

last_time = time.time()

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

def put_coord(startX,endX,startY,endY,IoU,ID,state,p_time):
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    info = {
        "startX": startX,
        "endX"  : endX,
        "startY": startY,
        "endY"  : endY,
        "IoU"   : IoU,
        "ID"    : ID,
        "State" : state,
        "Time"  : p_time
    }
    data["parking"].append(info)
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)

def cluster_parking ():
    global min_distance
    slot_box = []
    new_slot_box = []
    with open('dbScan_coordinates.json', 'r') as f:
        data = json.load(f)
    
    dataset=[]
    
    for element in data["vehicles"]:
        slot_box.append([element["startX"],element["endX"],element["startY"],element["endY"]])
        dataset.append([int(np.average([element["startX"],element["endX"]])),int(np.average([element["startY"],element["endY"]]))])
    
    #dataset.append([int(np.average([3173,3705])),int(np.average([1077,1465]))])
    new_array = np.array(dataset), None
    X, y = new_array
    dbscan = cluster.DBSCAN(eps=(min_distance*0.5),min_samples = 7).fit(X)
    print("XXXXXXXXXX : " , min_distance)
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    #print(labels)
    p_boxes = [None] * n_clusters_
    #print ("New empty list", p_boxes)
    for i in range(len(slot_box)):
        if labels[i] != -1:
            X1,X2,Y1,Y2 = slot_box[i]
            if p_boxes[labels[i]] != None:
                x1,x2,y1,y2 = p_boxes[labels[i]]
                p_boxes[labels[i]] = [X1+x1,X2+x2,Y1+y1,Y2+y2]
                new_slot_box.append(slot_box[i])
            else:
                p_boxes[labels[i]] = slot_box[i]
    #refresh_coord_dbScan(new_slot_box)
    new_labels = labels.tolist()
    #print(new_labels)
    for j in range(n_clusters_):
        num = new_labels.count(j)
        x1,x2,y1,y2 = p_boxes[j]
        p_boxes[j] = [int(x1/num),int(x2/num),int(y1/num),int(y2/num)]
    put_boxes(p_boxes)
    refresh_coord_dbScan(p_boxes)
    p_boxes.clear()
    slot_box.clear()
  
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show() 

def refresh_coord_dbScan(boxes):
    with open('dbScan_coordinates.json', 'w') as file:
        data = {"vehicles":[]}
        json.dump(data, file)
    for element in boxes:
        #print ("Vehicle box :")
        #print(element)
        X1 =int(element[0])
        X2 =int(element[1])
        Y1 =int(element[2])
        Y2 =int(element[3])
        info = {
            "startX": X1,
            "endX"  : X2,
            "startY": Y1,
            "endY"  : Y2
        }
        for i in range(7):
            data["vehicles"].append(info)
    with open('dbScan_coordinates.json', 'w') as file:
        json.dump(data, file)


def put_coord_dbScan(boxes):
    global min_distance
    global nSamples
    with open('dbScan_coordinates.json', 'r') as f:
        data = json.load(f)
    coor_array=[]
    for element in boxes:
        #print ("Vehicle box :")
        #print(element)
        X1,X2,Y1,Y2,v_type = element
        if v_type != "motorcycle" and v_type != "bus":
            info = {
                "startX": X1,
                "endX"  : X2,
                "startY": Y1,
                "endY"  : Y2
            }
            data["vehicles"].append(info)
            
            coor_array.append( [np.average ( element[:2] ), np.average ( element[2:4] ) ] )
    distance_array = distance.cdist(coor_array, coor_array, 'euclidean')
    temp_min = 10000
    for elements in distance_array:
        for item in elements:
            if item > 0 and item <= temp_min:
                temp_min = item
    if min_distance > temp_min:
        min_distance = temp_min
    #print(min_distance)
    with open('dbScan_coordinates.json', 'w') as file:
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
            i.append(None)
        
        put_coord( i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])



def sort_parking_id(boxes):
    
    sorted_boxes = sorted(boxes,key=lambda x : (x[0],x[2]))
    num = 1
    for i in sorted_boxes:
        length = len(i)
        if length == 5:
            i.append(num)
        else:
            i[5] = num
        num+=1
    return sorted_boxes

def load_coord():
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    slot_box =[]
    for element in data["parking"]:
        slot_box.append([element["startX"],element["endX"],element["startY"],element["endY"],element["IoU"],element["ID"],element["State"],element["Time"]])
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
    global dbscan_ready
    global last_time
    unlawful = [0] * len(v_boxes) 
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
        State = ""
        #print("P place coordinates are :",slot)
        print("Maximum IoU is: ",IoU_max)
        #print("Parking space IoU is: ",slot[4])
        
        if IoU_max < slot[4]+0.10:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),10)
           free_space += 1
           slot[6] = "Free"
        
        if IoU_max > slot[4]+0.10:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),10)
           slot[6] = "Busy"
           slot[7] = 0  if slot[7] is None else slot[7]+round((time.time()-last_time),3)

        cv2.putText(image, str(Id), (x1+50,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
    if dbscan_ready:
        for vehicle in v_boxes:
            x1 =int(vehicle[0])
            x2 =int(vehicle[1])
            y1 =int(vehicle[2])
            y2 =int(vehicle[3])
            v_type = vehicle[4]
            IoU_max = 0
            num_IoU =0
            for slot in p_boxes:
                IoU_temp = slot[4]
                IoU = calculate_IoU(slot,vehicle)
                if IoU > slot[4]+0.10 and  v_type == "car":
                    num_IoU += 1
            if (num_IoU > 1):
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,165,0),10)
                v_boxes.remove(vehicle)


    put_boxes(p_boxes)
    print("Number of free parking space is: ",free_space)
    print("Number of total parking space is: ",len(p_boxes))
    last_time = time.time()
    return free_space, v_boxes


def compare_spots(p_boxes,v_boxes):
    #print("Number of p boxes is:", len(p_boxes))
    #print("Number of v boxes is:", len(v_boxes))
    
    for vehicle in v_boxes:
        IoU_max = 0
        for park in p_boxes:
            IoU = calculate_IoU(park,vehicle)
            if IoU_max < IoU:
               IoU_max=IoU
               #print(park)
               x1,x2,y1,y2,temp_iou,Id,state,p_time = park
    
        X1,X2,Y1,Y2,v_type = vehicle
        if IoU_max < (temp_iou + 0.20) and dbscan_ready != True:
            #print("Temp IoU = ",temp_iou)
            print("Lägger till en ny p_plats")
            p_boxes.append([X1,X2,Y1,Y2])
            put_boxes(p_boxes)
            #print("New park space:", X1,X2,Y1,Y2)
            #print("IoU is: ", IoU_max )
        elif IoU_max > 0.80 and v_type != "motorcycle" and v_type != "bus" and dbscan_ready != True:
            print("Tar medelvärdet och uppdaterar, IoU is:", IoU_max)
            X1=(x1+X1)/2
            X2=(x2+X2)/2
            Y1=(y1+Y1)/2
            Y2=(y2+Y2)/2
            #Maybe add time here, reminder for myself!
            replace_coord(x1,y1,X1,X2,Y1,Y2)
                
        p_boxes.clear()
        p_boxes = load_coord()
    print("number OF TOTAL P PLACE : ", len(p_boxes))
    return p_boxes

def compare_boxes(BOXA,BOXB):
    return True if calculate_IoU(BOXA,BOXB)>0.9 else False


def compare_list (listA,listB):
    lengthA = len(listA)
    lengthB = len(listB)
    check =0
    for a in listA:
        for b in listB:
            if compare_boxes(a,b):
                check += 1
                break
    return True if (lengthA == check) and (lengthB == lengthA) else False




# |-------------------------------------------|
# |-----Functions for YOLO3------|
# |-------------------------------------------|

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, classify_treshold, nms_threshold)

    for i in indices:
        
        i = i[0]
        if(str(classes[class_ids[i]]) == "car" or str(classes[class_ids[i]]) == "truck" or str(classes[class_ids[i]]) == "motorcycle" or str(classes[class_ids[i]]) == "bus"):
            box = boxes[i]
            x1 = box[0]
            x2 = box[0]+box[2]
            y1 = box[1]
            y2 = box[3]+box[1]
            BOX.append([x1,x2,y1,y2,str(classes[class_ids[i]])])
    return BOX


#----------------------------------------------
# Check what mode is chosen,URL or local image-
#----------------------------------------------
#with open('coordinates.json', 'w') as file:
#        data = {"parking":[]}
#        json.dump(data, file)

print("Program starts")

while(True):
    try:
        x = int(input("Enter the image number from 1-7...")) #integer input
    except ValueError:
        x=1
    if(x<1 or x>7):
        print("Value out of bounds, choses img 1 instead")
        x=1

    image_yolo3 = copy(vars()["image"+str(x)])
    #image_yolo3 = get_cam_img()
    test_image = image_yolo3.copy()

    start = time.time()
    print("Loading coordinates")
    park_boxes = load_coord()
    #print("Json coordinats in the beginning", park_boxes)

    #image = get_cam_img()        

#-------------------------------------
# Methods for classifying the objects-
#-------------------------------------
    print("Classifying with Yolov3")
    vehicle_boxes = yolo3_classify(image_yolo3, classes, COLORS)
    vehicle_box_reserv = deepcopy(vehicle_boxes)
    if len(vehicle_boxes) > 0:
        print("Number of detected vehicle is: ", len(vehicle_boxes))
        while True:
            time.sleep(interval/2)
            print("Second round classifying just to be sure")
            vehicle_boxes2 = yolo3_classify(image_yolo3, classes, COLORS)
            if(compare_list(vehicle_boxes,vehicle_boxes2)):
                print("We are sure that at least one vehicle is there")
                break
            else: 
                print("We will classify again")
                vehicle_boxes = vehicle_boxes2
        if len(park_boxes) == 0:
            print("This is new program put every box as a parking space")
            park_boxes = deepcopy(vehicle_boxes)
            put_boxes(park_boxes)
        else:
            print("Comparing the spots")
            print(park_boxes)
            park_boxes = compare_spots(park_boxes,vehicle_boxes)

    print("Calculating free spots")
    new_free_spaces, vehicle_boxes = calculate_free_spots(image_yolo3,park_boxes,vehicle_boxes)
    

#-----------------
# Show the images-
#-----------------
    while((time.time()-start < interval) and (time.time()-start>0)):
        time.sleep(1)
        #print(time.time()-start)

    image_yolo3 = cv2.resize(image_yolo3, (800, 600))
    cv2.imshow("Parking Spaces", image_yolo3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    #print(park_boxes)
    if(free_spaces != new_free_spaces):
       lora_message = "New-"+str(ID)+"-"+str(len(park_boxes))+"-"+str(new_free_spaces)
       print(lora_message)
       #ser.write(lora_message.encode())
       #str1=ser.readline()
       #print(str1)
       free_spaces = new_free_spaces
       #if(len(vehicle_box_reserv)>0):
            #put_coord_dbScan(vehicle_box_reserv)
            #nSamples += 1
    if(len(vehicle_box_reserv)>0):
        put_coord_dbScan(vehicle_box_reserv)
        nSamples += 1
    vehicle_boxes.clear()
    if nSamples >= 3:
        cluster_parking()
        park_boxes = load_coord()
        for box in park_boxes:
            x1 =int(box[0])
            x2 =int(box[1])
            y1 =int(box[2])
            y2 =int(box[3])
            cv2.rectangle(test_image,(x1,y1),(x2,y2),(0,255,0),10)
        test_image = cv2.resize(test_image, (800, 600))
        cv2.imshow("New Parking Boxes", test_image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        nSamples = 0
        dbscan_ready = True
        #with open('dbScan_coordinates.json', 'w') as file:
            #data = {"vehicles":[]}
            #json.dump(data, file)
    vehicle_boxes.clear()
    park_boxes.clear()
