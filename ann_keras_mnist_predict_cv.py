import numpy as np
import cv2
import pickle
from skimage import img_as_ubyte		
from skimage.color import rgb2gray


w=640
h=480


cap=cv2.VideoCapture(0)
cap.set(3,w)
cap.set(4,h)

pickle_in=open("trained_model.p","rb")
model=pickle.load(pickle_in)



while(1):
    s,img=cap.read()
    im =np.asarray(img)
    cv2.imshow("Window", im)
    
    im_gray = rgb2gray(img)				#convert original to gray image
    
    img_gray_u8 = img_as_ubyte(im_gray)		# convert grey image to uint8
    
    #Convert grayscale image to binary
    (thresh, im_bw) = cv2.threshold(img_gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.imshow("Window", im_bw)
    #resize using opencv
    img_resized = cv2.resize(im_bw,(28,28))
    #cv2.imshow("Window", img_resized)
    #resize using sciikit
    #im_resize = resize(im,(28,28)')
    #plt.show()
    #cv2.imshow("Window", im_resize)
    
    im_gray_invert = 255 - img_resized
    im_final = im_gray_invert
    im=np.array([im_final])
    
    #print(im.shape)

    
    print("Number ",int(model.predict_classes(im)))

    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break

