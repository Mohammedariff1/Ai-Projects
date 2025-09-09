import cv2
#importing an opencv

alg = "haarcascade_frontalface_default.xml"
# Face Algorithm

haar_cascade = cv2.CascadeClassifier(alg)
cam = cv2.VideoCapture(0)
#Capturing a video

while True:
    _,img = cam.read()
    #cam.read() returns two values. One is true or false, another one is image.
    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #converting grayScale img for faster and low memory.
    
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    #A value of 1.3 means the image is reduced by 30% each time it’s scaled down.
    # 4 - Min neighbors – how many neighbors a rectangle should have to be called a face.
    # Getting an coordinates.
    
    for (x,y,w,h) in face:
        #x,y indicates top-left corner of rectangle
        #x+w,y+h indicates bottom-right corner of rectangle
        
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,255,0),5)
        
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    
    #Click esc key to exit
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
