import cv2
import numpy as np
import glob  
import sys
from imutils import perspective

from warp import warp
# Create point matrix get coordinates of mouse click on image
point_matrix = np.zeros((4,2),np.int)
 
counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        counter = counter + 1
        print(counter)
def empty(a):
    pass
imgs = glob.glob("new/input/*.jpg")
count = 1
thresh1 = 255
thresh2 = 255
thresh3 = 0 
thresh4 = 80 
thresh5 = 60 
thresh6 = 160 
# cv2.namedWindow("parameters")
# cv2.resizeWindow("parameters",640,240)
# cv2.createTrackbar("threshold1","parameters",255,255,(lambda a: None))
# cv2.createTrackbar("threshold2","parameters",255,255,(lambda a: None))
# cv2.createTrackbar("threshold3","parameters",150,255,(lambda a: None))


for i in glob.glob("warp\*.jpg"):
    try:  
        cv2.namedWindow("parameters")
        cv2.resizeWindow("parameters",640,240)
        cv2.createTrackbar("threshold1","parameters",thresh1,255,(lambda a: None))
        cv2.createTrackbar("threshold2","parameters",thresh2,255,(lambda a: None))
        cv2.createTrackbar("threshold3","parameters",thresh3,255,(lambda a: None))
        cv2.createTrackbar("threshold4","parameters",thresh4,255,(lambda a: None))
        cv2.createTrackbar("threshold5","parameters",thresh5,255,(lambda a: None))
        cv2.createTrackbar("threshold6","parameters",thresh6,255,(lambda a: None))
        boo=True
        orig = cv2.imread(i)
        height,width,_ = orig.shape

        
        while True:
            blur = cv2.GaussianBlur(orig.copy(),(1,1),1)
            
            thresh4 = cv2.getTrackbarPos("threshold4","parameters")
            thresh5 = cv2.getTrackbarPos("threshold5","parameters")
            thresh6 = cv2.getTrackbarPos("threshold6","parameters")
            img = blur.copy()
            value = thresh4
            
            img[img<thresh5] = 0
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayImage[grayImage<thresh6] = 0
            
            
            thresh3 = cv2.getTrackbarPos("threshold3","parameters")
            ret, thresh = cv2.threshold(grayImage, thresh3, 255, 0)
            
            thresh1 = cv2.getTrackbarPos("threshold1","parameters")
            thresh2 = cv2.getTrackbarPos("threshold2","parameters")
            

            #canny = cv2.Canny(thresh,thresh1,thresh2)
            canny2 = cv2.Canny(grayImage,thresh1,thresh2)
            kernal = np.ones((3,3))
            dil = cv2.dilate(canny2,kernal,iterations=1)
            
            conts = img.copy()
            max_area = 0
            contours,_ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            areas =[]
            for cnt in contours:
                area = cv2.contourArea(cnt)
                areas.append(area)
                '''
                if area>max_area:
                   max_area = area
                    c = cnt
                elif area<max_area and area>max_area2:
                    max_area2 = area
                    print("")
                    c2 = con
                '''
            cnts_index = np.argsort(areas)
            #print(cnts_index)

            c = contours[cnts_index[-1]]
            i = 2
            while True:
                if len(cnts_index)>=i and cv2.contourArea(c)<(height*width*0.4):
                    c2 = contours[cnts_index[-i]] 
                    #cv2.drawContours(conts,c2,-1,(255,0,0),2)
                    c = np.concatenate((c,c2),axis=0)
                    i+=1
                else:break
            cv2.drawContours(conts,c,-1,(255,0,255),2)
              
            
            
            #if boo:
            rect = cv2.minAreaRect(c)
            #print(rect)
            box = cv2.boxPoints(rect)
            #print(box)
            box_ = perspective.order_points(box)
            out = warp(img,box_)
            cv2.imshow("warped",out)
            
            cv2.imshow("orig",img)
            cv2.imshow("dil",dil)
            cv2.imshow("canny2",canny2)
            cv2.imshow("gray",grayImage)
            cv2.imshow("conts",conts)
            
            cv2.waitKey(1)
    except KeyboardInterrupt:
        sys.exit()
    except:
        
        print(sys.exc_info())
        pass