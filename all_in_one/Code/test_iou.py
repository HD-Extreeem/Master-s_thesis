import numpy as np
import cv2

BOXA = [2,8,2,8]

#BOXB = [20,80,20,80]
BOXB = [5,11,5,11]

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
    print(iou)
    return iou



# Create a black image
#img = np.zeros((512,512,3), np.uint8)
#cv2.rectangle(img,(20,20),(80,80),(0,255,0),3)
#cv2.rectangle(img,(50,50),(110,110),(0,0,255),3)
#cv2.imshow("IoU", img)
#cv2.waitKey()
calculate_IoU(BOXA,BOXB)
#print(intersection(BOXB,BOXA))
#print(union(BOXA,BOXB))
