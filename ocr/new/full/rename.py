import glob
import cv2
import numpy as np 
import matplotlib.pyplot as plt
#from tesseract_test import warp_plate
files = glob.glob("*.txt")

print(files)
input()
count = 1
#get rid of small dots
for file in files:
    try:
        
        img = cv2.imread(file[:-3]+"jpg")
        dw,dh,_ = img.shape
        
        #sr = cv2.dnn_superres.DnnSuperResImpl_create()

        #path = "FSRCNN_x3.pb"

        # sr.readModel(path)

        # sr.setModel("fsrcnn",3)

        # result = sr.upsample(img)
        # resized = cv2.resize(img,dsize=None,fx=3,fy=3)
        # plt.figure(figsize=(12,8))
        # plt.subplot(1,3,1)# Original image
        # plt.imshow(img[:,:,::-1])
        # plt.subplot(1,3,2)
        # # SR upscaled
        # plt.imshow(result[:,:,::-1])
        # plt.subplot(1,3,3)
        # # OpenCV upscaled
        # plt.imshow(resized[:,:,::-1])
        # plt.show()


        # cv2.imshow("1",img)
        # cv2.imshow("res",result)
        # cv2.waitKey()
        
        #grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #dst_rbg = cv2.Canny(orig, 75, 200)
        #ret, thresh = cv2.threshold(grayImage, 127, 255, 0)
        #resized = cv2.resize(thresh,(25,25))
        #invert = cv2.bitwise_not(thresh)
        #find all your connected components (white blobs in your image)
        #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(invert, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        #sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        #min_size = 75

        #your answer image
        #img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        # for i in range(0, nb_components):
            # if sizes[i] >= min_size:
                # img2[output == i + 1] = 255
        # out = 255-img2
        #cv2.imshow("2",out)
        # cv2.imshow("1",img2)
        #cv2.waitKey()
        
        #cv2.imwrite(file,out)
        
        with open(file[:-3]+"txt","r") as f:
            t = f.read().split("\n")
            bbs = [list(map(float,i.split())) for i in t[:-1]]
        
        #bbs.extend([list(map(float,j)) for j in temp])
        
        _,x,y,w,h = bbs[0]
        #print((x - (w / 2)) * dw,y,w,h)
        l = int((x - (w*1.0 / 2)) * dh)
        r = int((x + (w*1.0 / 2)) * dh)
        t = int((y - h*1.0/2) * dw)
        b = int((y + (h*1.0 / 2)) * dw)
        #print(l,r,t,b)

        
        cropped = img[t:b,l:r]
        #warped = warp_plate(cropped)
        #cv2.rectangle(img, (l+2,t+2), (r-2,b-2), (255,0,0), 2)
        cv2.imshow("v",cropped)
        
        print(str(count)+"_.jpg", cropped.shape)
        
        cv2.imwrite("front/"+str(count)+".jpg",cropped)
        count+=1
    except:pass
    
    