import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import time
from imutils import perspective
import sys
import imutils

def warp(img,box):
    #print(np.round(box))
    pt_A,pt_D,pt_C,pt_B = box
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))
     
     
    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
                            
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    #resized = cv2.resize(out,(maxWidth*2,maxHeight*2))
    #cv2.imshow("out",resized)
    #cv2.waitKey()
    return out

def detect_tilt(dst,orig):
    # get largest contour, use top two points as reference for rotation
    # pass canny
    
    
    contours,_ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recs = []
    areas = {}
    max_area = 0
    c = None
    height,width = dst.shape
    areas =[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    cnts_index = np.argsort(areas)
    
    c = contours[cnts_index[-1]]
    i = 2
    while True:
        if len(cnts_index)>=i and cv2.contourArea(c)<(height*width*0.35):
            c2 = contours[cnts_index[-i]] 
            c = np.concatenate((c,c2),axis=0)
            i+=1
        else:break
    conts = orig.copy()
    cv2.drawContours(conts,c,-1,(255,0,255),2)  
    fig.add_subplot(rows, columns,6)
    plt.imshow(conts, cmap="gray")
    plt.title("contours")
    # c = max(edge_contours)
    #x, y, w, h = cv2.boundingRect(c)
    
    # calculate angle to rotate image
    
    rect = cv2.minAreaRect(c) # rect[2] contains angle
    box = cv2.boxPoints(rect)
    
    
    y_sorted = box[np.argsort(box[:, 1]), :]
    tp_cs = np.sort(y_sorted[2:], axis=0)
    angle = abs(rect[-1])
    
    
    box_ = perspective.order_points(box)
    return (angle, c,box_)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def warp_plate(orig,gray_thresh = 150,angle_thresh = 10):
    
    #print(orig.shape)

    img = orig.copy()
    #img[img<60]=0
    #brightness
    value = 75
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_hsv[final_hsv<60]=0
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    #blur
    blur = cv2.GaussianBlur(orig.copy(),(1,1),1)
    
    grayImage = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grayImage[grayImage<120] = 0
    fig.add_subplot(rows, columns,2)
    plt.imshow(grayImage, cmap="gray")
    plt.title("gray")
    
    #ret, thresh = cv2.threshold(grayImage, 120, 255, 0)
    # fig.add_subplot(rows, columns,8)
    # plt.imshow(thresh, cmap="gray")
    # plt.title("thresh")
    
    
    canny = cv2.Canny(grayImage, 255, 255)
    fig.add_subplot(rows, columns,3)
    plt.imshow(canny, cmap="gray")
    plt.title("canny")
    
    kernal = np.ones((3,3))
    dil = cv2.dilate(canny,kernal,iterations=1)
    fig.add_subplot(rows, columns,4)
    plt.imshow(dil, cmap="gray")
    plt.title("dilated")
    
    angle, c,box = detect_tilt(dil,orig)
    fig.add_subplot(rows, columns,5)
    plt.imshow(orig, cmap="gray")
    plt.title("orig after")
    

    
    #print(angle)    
    #detected angle being more than 60-70 degrees is not considered
    
    #if abs(angle)>-1 and abs(angle)<80:
    out = warp(orig,box)
    fig.add_subplot(rows, columns,7)
    plt.imshow(out, cmap="gray")
    plt.title("out")
    #cv2.imshow("c",thresh)
    #print(angle,box[i])
    #cv2.waitKey()
    
    return out
    #return orig

if __name__ == '__main__':
    imgs = glob.glob("D:\Internship\warp\*.jpg")
    print(len(imgs))
    for img in imgs:
        print(img)
        try:
            fig = plt.figure(figsize=(6, 6))
            columns = 3
            rows = 3
            img = cv2.imread(img)
            fig.add_subplot(rows, columns,1)
            plt.imshow(img, cmap="gray")
            plt.title("img")
            out= warp_plate(img)
            # cv2.imshow("orig",img)
            plt.show()
            # cv2.imshow("out",out)
            # cv2.waitKey()
        except KeyboardInterrupt:sys.exit()
        except:
            print(sys.exc_info())
            pass