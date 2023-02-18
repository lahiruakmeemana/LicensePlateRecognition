from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import glob
from warp import warp_plate
file = ""
imgs = glob.glob("D:\Internship\sl_plate\*.jpg")
print(len(imgs))


# Write output file


net_plate = cv2.dnn.readNetFromDarknet('yolov3_plates.cfg','yolov3_plates_final.weights')
net_front = cv2.dnn.readNetFromDarknet('yolov3_front.cfg','yolov3_front_last.weights')
net_ocr = cv2.dnn.readNetFromDarknet('ocr/ocr-net_old.cfg','ocr/ocr-net_old.weights')
net_plate.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_plate.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net_front.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_front.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net_ocr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_ocr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

front_classes = ["front",""]
plate_classes = ["plate",""]
ocr_classes = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
#cap = 'test_images/<your_test_image>.jpg'
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

def predict(img,net,classes,size,thresh,name,rect=False,scale=False):
    height, width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img,1/255,size, (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    s = time.time()
    layerOutputs = net.forward(output_layers_names)
    print(name," time: ",time.time()-s)

    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > thresh:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    if scale:
                        w=int(w*1.2)
                        h=int(h*1.2)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    out_boxes = []
    out_classes = []
    #print("confidense: ",confidences)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            out_boxes.append(boxes[i])
            out_classes.append(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = (colors[i])
            
            if rect:cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            #cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
    return img,out_boxes,out_classes
val = 0    
for file in imgs:
    print(file)
    orig = cv2.imread(file)
    img,boxes,_ = predict(orig,net_front,front_classes,(288,288),0.4,"Front",True,True)
    
    for box in boxes:
        x,y,w,h = box
        plate,plate_box,_ = predict(img[y:y+h,x:x+w],net_plate,plate_classes,(416,416),0.6,"Plate",False,True)
        
        if len(plate_box)!=0:
            plate_box = np.array(plate_box[0])
            plate_box[plate_box<0] = 0
            px1,py1,pw,ph = plate_box
            
            px2 = px1+pw
            py2 = py1+ph
            
            
            if py2>h:py2 = h
            if px2>w:px2 = w
            #plate[py:py+ph,px:px+pw]
            
            cv2.imwrite("new/"+str(val)+".jpg",plate[py1:py2,px1:px2])
            val+=1
            warped = warp_plate(plate[py1:py2,px1:px2])
            ocr_img,letter_boxes,letter_classes = predict(warped,net_ocr,ocr_classes,(240,80),0.5,"OCR")
            area_thresh = round(0.65 * sum([i[2]*i[3] for i in letter_boxes])/len(letter_boxes))
            plate_num = ''
            try:
                order = np.argsort(letter_boxes,axis=0)[:,0]
                letter_classes = np.array(letter_classes)[order]
                print(letter_classes)
                for j,i in enumerate(np.array(letter_boxes)[order]):
                    print(letter_classes[j],i[2]*i[3])
                    if i[2]*i[3] >area_thresh:
                        
                        plate_num += letter_classes[j]
                if len(plate_num)==6:index=2
                elif len(plate_num)==7:index=3
                else:
                    print("not a valid plate number ",plate_num)
                    continue
                if '0' in plate_num[:index]:
                    i = plate_num.index('0')
                    plate_num = plate_num[:i]+'O'+plate_num[i+1:]
                print(plate_num)
                
            except:pass
        img[y:y+h,x:x+w] = plate
    #cv2.imshow("original",orig)    
    cv2.imshow("1",warped)#cv2.resize(img,(416,416)))
    cv2.waitKey()
    print()