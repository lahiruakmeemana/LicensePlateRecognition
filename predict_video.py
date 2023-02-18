from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from warp import warp_plate
import sys
cap = cv2.VideoCapture("video1.mp4")

# Write output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fps = 30
size = (int(frame_width),int(frame_height))

t1 = time.time()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("result2.mp4", fourcc, 30,(frame_width, frame_height), True)
frame_count = 0

net_plate = cv2.dnn.readNetFromDarknet('yolov3_plates.cfg','yolov3_plates_last.weights')
net_front = cv2.dnn.readNetFromDarknet('yolov3_front.cfg','yolov3_front_last.weights')
net_ocr = cv2.dnn.readNetFromDarknet('ocr/ocr-net.cfg','ocr/ocr-net.weights')
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

def predict(img,net,classes,size,thresh,rect=False,scale=False):
    height, width,_ = img.shape

    blob = cv2.dnn.blobFromImage(img,1/255,size, (0,0,0), swapRB=True, crop=False)
    
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    s = time.time()
    layerOutputs = net.forward(output_layers_names)
    print("time: ",time.time()-s)

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
                        w=int(w*1.1)
                        h=int(h*1.1)
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

while True:
    ret, next_frame = cap.read() # Reads the next video frame into memory

    if ret == False: break
    frame_count += 1
    if frame_count%1==0:
        try:
            img,boxes,_ = predict(next_frame,net_front,front_classes,(288,288),0.4,True)
            for box in boxes:
                x,y,w,h = box
                plate,plate_box,_ = predict(img[y:y+h,x:x+w],net_plate,plate_classes,(416,416),0.6,True,True)
                #print(plate_box)
                
                if len(plate_box)!=0:
                    plate_box = np.array(plate_box[0])
                    px,py,pw,ph = plate_box
                    plate_box[plate_box<0] = 0
                    
                    if plate_box[0]>w:plate_box[0]=w
                    if plate_box[1]>h:plate_box[1] = h
                    #print(plate_box)
                    #cv2.imshow("plate",plate[plate_box[1]:plate_box[1]+plate_box[3],plate_box[0]:plate_box[0]+plate_box[2]])
                    #warped = warp_plate(plate[plate_box[1]:plate_box[1]+plate_box[3],plate_box[0]:plate_box[0]+plate_box[2]])
                    try:    
                        ocr_img,letter_boxes,letter_classes = predict(plate[py:py+ph,px:px+pw],net_ocr,ocr_classes,(240,80),0.5)
                    
                        order = np.argsort(letter_boxes,axis=0)[:,0]
                    
                        print(np.array(letter_classes)[order])
                    except:pass
                img[y:y+h,x:x+w] = plate
                        
                # write frame
                cv2.imshow("1",img)
                writer.write(img)

                key = cv2.waitKey(50)

                if key == 27: # Hit ESC key to stop
                    break
        except:
            if KeyboardInterrupt:sys.exit()
            pass
        

# end timer
t2 = time.time()

# calculate FPS
fps = str( float(frame_count / float(t2 - t1))) + ' FPS'

print("Frames processed: {}".format(frame_count))
print("Elapsed time: {:.2f}".format(float(t2 - t1)))
print("FPS: {}".format(fps))

cap.release()
cv2.destroyAllWindows()
writer.release()
