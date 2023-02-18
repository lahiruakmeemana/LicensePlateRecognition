import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
#import pytesseract
import glob
import time
from  scipy import ndimage
from imutils import perspective

def build_tesseract_options(psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        options += " --oem {}".format(3)
        
        # return the built options string
        return options

def warp(img,box):
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
    resized = cv2.resize(out,(maxWidth*2,maxHeight*2))
    #cv2.imshow("out",resized)
    #cv2.waitKey()
    return out

def detect_tilt(dst,orig):
    # get largest contour, use top two points as reference for rotation
    # pass canny
    edge_contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    recs = []
    areas = {}
    max_area = 0
    c = None
    height,width = dst.shape
    for con in edge_contours:
        x,y,w,h = cv2.boundingRect(con)
        
        # if ( ((3 * h) > w) and (w > (1.7 * h))):
        if (w > (1.5 * h)):
            area = w * h
            if area > max_area:
                #print("updating max area")
                max_area = area
                c = con
               
    # c = max(edge_contours)
    x, y, w, h = cv2.boundingRect(c)
    
    # calculate angle to rotate image
    rect = cv2.minAreaRect(c) # rect[2] contains angle
    box = cv2.boxPoints(rect)
    #box_ = perspective.order_points(box)
    
    # loop over the original points
    for i in range(len(box)):
        box[box<0] = 0
        if box[i][0]>width:box[i][0]=width
        if box[i][1]>height:box[i][1] = height
        # draw circles corresponding to the current points and
        #cv2.circle(orig, (int(box[i][0]), int(box[i][1])), 1, (0,0,255), -1)
        

   
    
    
    # sort by y values
    y_sorted = box[np.argsort(box[:, 1]), :]
    tp_cs = np.sort(y_sorted[2:], axis=0)
    angle = np.rad2deg(np.arctan2(tp_cs[1][1] - tp_cs[0][1], tp_cs[1][0] - tp_cs[0][0]))
    # if tilted right, rotate left (counter-clockwise)
    #print(x,y,w,h,rect,box)
    if y_sorted[2:][0][0] > y_sorted[2:][1][0]:
        angle = - angle
    # rotate image
    box_ = perspective.order_points(box)
    return (angle, c,box_)
def trim_border(image):
    '''
    thin out border
    white out lines that are > 90% black (in case of plate that touches border).
    high mean implies row/column is mostly white.
    '''
    np.seterr(divide='ignore', invalid='ignore')
    columns_mean = np.mean(image, axis = 0)
    rows_mean = np.mean(image, axis = 1)
    # whiteout rows and columns that are mostly black, assuming those are borders
    row_border_threshold = 10
    column_border_threshold = 25
    border_rows = np.where(rows_mean < row_border_threshold )
    border_columns = np.where(columns_mean < column_border_threshold )
    # whiteout rows and columns that are mostly black, assuming those are borders
    image[[border_rows], :] = 255
    image[:, [border_columns]] = 255
    return image

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def run(img):
    files = glob.glob(r"D:\Internship\sl*")
    print(len(files))
    lp = []
    s = time.time()
    #for file in files:
    #filename = "D:\Internship\dataset\selected\plates\carsgraz_241_0.jpg"
    pytesseract.tesseract_cmd = r"C:\Users\Lahiru\AppData\Local\Tesseract-OCR\tesseract.exe"
    options = build_tesseract_options(psm=7)
    #img = cv2.imread(file)
    #rotated = rotate_image(img,15)
    #gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    # threshold the image using Otsus method to preprocess for tesseract
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform a median blur to smooth image slightly
    #blur = cv2.medianBlur(gray, 3)
    # resize image to double the original size as tesseract does better with certain text size
    #blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow("blur",img)
    
    text = pytesseract.image_to_string(img,config=options) 
    lp.append([text.split('\n'),img.shape])
    print(time.time()-s)
    for i in lp:
        print(i)
    cv2.waitKey()

def max_line(lines):
    index = 0
    length = 0
    for i,line in enumerate(lines):
        if (line[0][0]-line[0][2])>length:
            index = i 
    return index


def warp_plate(orig):
    
    #print(orig.shape)

    grayImage = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    #dst_rbg = cv2.Canny(orig, 75, 200)
    ret, thresh = cv2.threshold(grayImage, 127, 255, 0)
    ret, threshbin = cv2.threshold(grayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    dst = cv2.Canny(thresh, 75, 200)


    ret, threshcanny = cv2.threshold(dst, 150, 255, 0)

    angle, c,box = detect_tilt(thresh,orig)
    cv2.drawContours(grayImage,c,-1,(0,255,0),3)
    out = warp(orig,box)
    #cv2.imshow("c",threshcanny)
    #print(angle,box)
    #cv2.waitKey()
    
    return out

if __name__ == '__main__':
    img = cv2.imread("plate.jpg")
    out= warp_plate(img)
    cv2.imshow("out",out)
    cv2.waitKey()