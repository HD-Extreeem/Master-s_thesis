# -----------------------------------------------------------------|
#   TO RUN TYPE:                                                   |
#   python smart_parking_v9.py
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

#Serial com to the badgerboard
#Configured statically
ser = serial.Serial('/dev/badgerboard',9600,5)

#Url to the camera capture
url_img = 'http://root:ateapass@192.168.0.90/axis-cgi/jpg/image.cgi?resolution=1920x1080'

ID                = 1
interval          = 60*5
dbscan_ready      = False
min_distance      = 10000
nSamples          = 0
free_spaces       = -1
vehicle_boxes     = []
old_vehicle_boxes = []
park_boxes        = []
Min_Samples       = 180
classify_treshold = 0.48
classes           = None
last_time         = time.time()

# Requires to download the yolo lable file from
# https://pjreddie.com/darknet/yolo/
# And place in the yolo folder from the current folder as the code
with open("yolo3/yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# |----------------------------------------------------|
# |-Method that reads the distance from the json file -|
# |----------------------------------------------------|
def read_distance():
    distance = -1
    with open('distance.json', 'r') as f:
        data = json.load(f)
    for element in data["distance"]:
        distance = element["min"]
    return distance

# |-------------------------------------------------------------------------|
# |-Writes the distance to the json file in order to store the min distance-|
# |-------------------------------------------------------------------------|
def write_distance(new_dist):
    with open('distance.json', 'w') as file:
        data = {"distance":[]}
        json.dump(data, file)
    info = {"min":new_dist}
    data["distance"].append(info)
    with open('distance.json', 'w') as file:
        json.dump(data, file)

# |----------------------------------------------|
# |-This method request an image from the camera-|
# |----------------------------------------------|
def get_cam_img():
    r = requests.get(url_img)
    print("Status code"+str(r.status_code))
    if r.status_code == 200:
        image = np.asarray(bytearray(r.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image

# |-------------------------------------------------|
# |-This method appends the coordinates            -|
# |-and its neccessary information to the json file-|
# |-------------------------------------------------|
def put_coord(startX,endX,startY,endY,IoU,ID,state):
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    info = {
        "startX": startX,
        "endX"  : endX,
        "startY": startY,
        "endY"  : endY,
        "IoU"   : IoU,
        "ID"    : ID,
        "State" : state
    }
    data["parking"].append(info)
    with open('coordinates.json', 'w') as file:
        json.dump(data, file)

# |----------------------------------------------------------------------------|
# |-This method performs the clustering of the collected coordinates          -|
# |-Clusters the coordinates by taking the centroid of the coordinates        -|
# |-When the clustering is finished, the outliers are removed if thera are any-|
# |----------------------------------------------------------------------------|
def cluster_parking ():
    global Min_Samples
    distance = int(read_distance())
    slot_box = []
    new_slot_box = []
    with open('dbScan_coordinates.json', 'r') as f:
        data = json.load(f)
    
    dataset=[]
    
    for element in data["vehicles"]:
        slot_box.append([element["startX"],element["endX"],element["startY"],element["endY"]])
    dataset.append([int(np.average([element["startX"],element["endX"]])),int(np.average([element["startY"],element["endY"]]))])
    
    new_array = np.array(dataset), None
    X, y = new_array

    # Clusters the data
    dbscan = cluster.DBSCAN(eps=(distance*0.4),min_samples = 7).fit(X)
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    p_boxes = [None] * n_clusters_

    #Loop through the data and keep only the cluster in order to remove outliers
    for i in range(len(slot_box)):
        if labels[i] != -1:
            X1,X2,Y1,Y2 = slot_box[i]
            if p_boxes[labels[i]] != None:
                x1,x2,y1,y2 = p_boxes[labels[i]]
                p_boxes[labels[i]] = [X1+x1,X2+x2,Y1+y1,Y2+y2]
                new_slot_box.append(slot_box[i])
            else:
                p_boxes[labels[i]] = slot_box[i]
    new_labels = labels.tolist()

    #Loop through the valid clusters and take the average of each parking spots
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

    #Loop thorugh the data and creates a visual plot of the clusters
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

# |-----------------------------------------------------------------------|
# |-This method stores the valid coordinates from the clustering task    -|
# |-The coordinates are stored in a seperate json file for the clustering-|
# |-----------------------------------------------------------------------|
def refresh_coord_dbScan(boxes):
    with open('dbScan_coordinates.json', 'w') as file:
        data = {"vehicles":[]}
        json.dump(data, file)
    for element in boxes:
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
        for i in range(int(Min_Samples/2)):
            data["vehicles"].append(info)
    with open('dbScan_coordinates.json', 'w') as file:
        json.dump(data, file)

# |--------------------------------------------------------------------|
# |-Method for appending the parking spot coordinates to the json file-|
# |-These coordinates are used for the clustering task -|
# |--------------------------------------------------------------------|
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
        write_distance(min_distance)
    with open('dbScan_coordinates.json', 'w') as file:
        json.dump(data, file)

# |--------------------------------------------------------------------|
# |-Method for appending the parking spot coordinates to the json file-|
# |--------------------------------------------------------------------|
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
        put_coord( i[0], i[1], i[2], i[3], i[4], i[5], i[6])


# |----------------------------------------------------|
# |-Method for sorting the parking spots in order     -|
# |-The parking spots are ordered by their coordinates-|
# |----------------------------------------------------|
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

# |-----------------------------------------------------------------------------------|
# |--Method that reads the stored coordinate from the json file-----------------------|
# |-----------------------------------------------------------------------------------|
def load_coord():
    with open('coordinates.json', 'r') as f:
        data = json.load(f)
    slot_box =[]
    for element in data["parking"]:
        slot_box.append([element["startX"],element["endX"],element["startY"],element["endY"],element["IoU"],element["ID"],element["State"]])
    return slot_box

# |---------------------------------------------------------------------------------------------|
# |- Method replaces the stored coordinates with the new average value that has been caluclated-|
# |---------------------------------------------------------------------------------------------|
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

# |-----------------------------------------------------------------------------------|
# |- Method that calculates the intersection of two different lists of bounding boxes-|
# |- Returns the intersection value of both boxes                                    -|
# |-----------------------------------------------------------------------------------|
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

# |-----------------------------------------------------------------------|
# |- Method calculates the IoU value of the vehicle with the bounding box-|
# |- Also appends the maximum intersection with vehicles that are close  -|
# |-----------------------------------------------------------------------|
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

# |-------------------------------------------------------------------|
# |- Method that checks which spots are busy/free on the parking lot -|
# |- Also checks if there are vehicles that are parked unlawfully    -|
# |-------------------------------------------------------------------|
def calculate_free_spots(image,p_boxes,v_boxes):
    free_space=0
    global dbscan_ready
    global last_time
    
    # Loop through all the stored parking bounding boxes and detected vehicle bounding boxes
    # And Check if they parking spots are busy or free
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

        print("Maximum IoU is: ",IoU_max)

        # If the IoU of the parking spot is smaller than the threshold, then we draw a green box
        if IoU_max < slot[4]+0.10:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),10)
           free_space += 1
           slot[6] = "Free"
               
        # else if it is greater than the threshold
        # then we draw a red box to indicate the parking spot as busy
        elif IoU_max > slot[4]+0.50:
           cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),10)
           slot[6] = "Busy"

        # Else it might be that the vehicle is parked unlawfully
        else:
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),10)

        cv2.putText(image, str(Id), (x1+10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4)

    # Checks only when the clustering is made
    # Checks if any vehicle is parked unlawfully
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
            if (num_IoU > 1) or (num_IoU==0):
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,165,0),10)


    put_boxes(p_boxes)
    print("Number of free parking space is: ",free_space)
    print("Number of total parking space is: ",len(p_boxes))
    last_time = time.time()
    return free_space, v_boxes

# |-------------------------------------------------------------------|
# |- Method that compares the already stored parking spot coordinates-|
# |- and the new parking spot detection                              -|
# |-------------------------------------------------------------------|
def compare_spots(p_boxes,v_boxes):
    
    for vehicle in v_boxes:
        IoU_max = 0
        # Loop through the new predicted bounding boxes and already stored bounding boxes
        # Calculates the IoU value and stores the maximum Iou and the parking information
        for park in p_boxes:
            IoU = calculate_IoU(park,vehicle)
            if IoU_max <= IoU:
               IoU_max=IoU
               x1,x2,y1,y2,temp_iou,Id,state = park
    
        X1,X2,Y1,Y2,v_type = vehicle
        # If the IoU is very small then we store the coordinate
        # Since this is a new possible parking spot
        if IoU_max < (temp_iou + 0.20) and dbscan_ready != True:
            print("Lägger till en ny p_plats")
            p_boxes.append([X1,X2,Y1,Y2])
            put_boxes(p_boxes)
        
        # Else if the intersection is greater than 80% and not a bus/motorcycle
        # Then we take the average value of the parking spot (Coordinate)
        elif IoU_max > 0.80 and v_type != "motorcycle" and v_type != "bus" and dbscan_ready != True:
            print("Taking the average value and updating, Iou is: ", IoU_max)
            X1=(x1+X1)/2
            X2=(x2+X2)/2
            Y1=(y1+Y1)/2
            Y2=(y2+Y2)/2
            replace_coord(x1,y1,X1,X2,Y1,Y2)
                
        p_boxes.clear()
        p_boxes = load_coord()
    print("number OF TOTAL P PLACE : ", len(p_boxes))
    return p_boxes

# |--------------------------------------------------------------------|
# |-Method that compares the two bounding boxes by calculating the IoU-|
# |--------------------------------------------------------------------|
def compare_boxes(BOXA,BOXB):
    return True if calculate_IoU(BOXA,BOXB)>0.9 else False


# |-------------------------------------------------------------------|
# |-Compares two different list to check if they are almost identical-|
# |-------------------------------------------------------------------|
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
# |-----------Functions for YOLO3-------------|
# |-------------------------------------------|

# Method that get the result of the predictions from the classification
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

#Method that classify and filter the img processing result to store only vehicles
def yolo3_classify(image_yolo, classes, COLORS):
    Width = image_yolo.shape[1]
    Height = image_yolo.shape[0]
    scale = 0.00392
    
    # Requires to download the yolo config and weight file from
    # https://pjreddie.com/darknet/yolo/
    # And place in the yolo folder from the current folder as the code
    net = cv2.dnn.readNet("yolo3/yolov3.cfg", "yolo3/yolov3.weights")
    
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

    #Loops through the detected objects and stores only the vehicles
    for i in indices:
        i = i[0]
        if(str(classes[class_ids[i]]) == "car" or str(classes[class_ids[i]]) == "truck" or str(classes[class_ids[i]]) == "motorcycle" or str(classes[class_ids[i]]) == "bus"):
            box = boxes[i]
            x1 = box[0]
            x2 = box[0]+box[2]
            y1 = box[1]
            y2 = box[3]+box[1]
            BOX.append([x1,x2,y1,y2,str(classes[class_ids[i]])])
            print("CONFIDENCES", confidences[i])
    return BOX

while(True):
    #Capture image from the camera
    image_yolo3 = get_cam_img()

    start = time.time()
    print("Loading coordinates")
    park_boxes = load_coord()
    min_distance = int (read_distance())
    
#-------------------------------------
# Methods for classifying the objects-
#-------------------------------------
    print("Classifying with Yolov3")
    vehicle_boxes = yolo3_classify(image_yolo3, classes, COLORS)
    vehicle_box_reserv = deepcopy(vehicle_boxes)
    
    #Perform only calculations if there are vehicles that are present on the parking lot
    if len(vehicle_boxes) > 0:
        print("Number of detected vehicle is: ", len(vehicle_boxes))
        #Run until the classification and parking coordinates match
        #This in order to make sure that vehicles are not being moved while predicting
        while True:
            time.sleep(interval/2)
            image_yolo3 = get_cam_img()
            print("Second round classifying just to be sure")
            vehicle_boxes2 = yolo3_classify(image_yolo3, classes, COLORS)
            if(compare_list(vehicle_boxes,vehicle_boxes2)):
                print("We are sure that at least one vehicle is there")
                test_image = deepcopy(image_yolo3)
                break
            else: 
                print("We will classify again")
                vehicle_boxes = vehicle_boxes2
    
        #Run this statement if this program was run the first time
        if len(park_boxes) == 0:
            print("This is new program put every box as a parking space")
            park_boxes = deepcopy(vehicle_boxes)
            put_boxes(park_boxes)
                
        #Else, there are already parking spaces defined
        else:
            print("Comparing the spots")
            print(park_boxes)
            park_boxes = compare_spots(park_boxes,vehicle_boxes)

    #Check if any free spots and if there are any new parking spots
    new_free_spaces, vehicle_boxes = calculate_free_spots(image_yolo3,park_boxes,vehicle_boxes)
    

#-----------------
# Show the images-
#-----------------

    while((time.time()-start < interval) and (time.time()-start>0)):
        time.sleep(1)

    image_yolo3 = cv2.resize(image_yolo3, (1000, 700))
    cv2.imshow("Parking Spaces", image_yolo3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("test-image.jpg",image_yolo3)
    new_list =[]

    #Check whether the same vehicles are parked and no changes have been made on the parking lot
    if(free_spaces != new_free_spaces):
       #Compose the lora message to send
       lora_message = "New-"+str(ID)+"-"+str(len(park_boxes))+"-"+str(new_free_spaces)
       print(lora_message)
       ser.write(lora_message.encode())
       #str1=ser.readline()
       #print(str1)
       free_spaces = new_free_spaces
    #Statement that checks if there are any parked vehicles
    #If there are vehicles parked, then compare if it is the same from the last check
    #Increase sample count if there is a new vehicle that have parked
    if(len(vehicle_box_reserv)>0):
        for j in vehicle_box_reserv:
            for k in old_vehicle_boxes:
                if (compare_boxes(j,k) != True):
                    new_list.append(j)
                    nSamples += 1
                    break
        if(len(new_list)>0):
            put_coord_dbScan(vehicle_box_reserv)
            
    print("Number of total new detected vehicles is :", len(new_list))
    old_vehicle_boxes = deepcopy(vehicle_boxes)
    vehicle_boxes.clear()
    #Check if we have collected enough samples for performing the clustering task
    if nSamples >= Min_Samples:
        cluster_parking()
        park_boxes = load_coord()
        test_image = deepcopy(image100)
        for box in park_boxes:
            x1 =int(box[0])
            x2 =int(box[1])
            y1 =int(box[2])
            y2 =int(box[3])
            cv2.rectangle(test_image,(x1,y1),(x2,y2),(0,255,0),10)
        test_image = cv2.resize(test_image, (1000, 700))
        cv2.imshow("New Parking Boxes", test_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite("New-Boxes_Clustring.jpg",test_image)
        nSamples = 0
        dbscan_ready = True

        # Clears the collected data for clustering
        #with open('dbScan_coordinates.json', 'w') as file:
            #data = {"vehicles":[]}
            #json.dump(data, file)

    vehicle_boxes.clear()
    park_boxes.clear()
    print("Number of samples is : ", nSamples)
