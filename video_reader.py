import cv2
import numpy as np

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
a=1
first_frame= None
crop_img = None
while True:
    a=a+1
    check,frame = video.read()
    key = cv2.waitKey(1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    gray.astype(np.float32)
    #face= face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10)
    
    #loop for  first frame
    if first_frame is None:
        first_frame = gray
        face= face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10)
        for x,y,w,h in face:
            crop_img = first_frame[y:y+h , x:x+w]
            w, h = crop_img.shape[::-1]
        continue
    
    if crop_img is None:
        print("Unable to detect face")
        first_frame = None
        continue
    
    res=cv2.matchTemplate(gray,crop_img ,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    frame=cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2)
    
    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255,cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame,None, iterations=2)
    
    
    cv2.imshow("frame",frame)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("crop", crop_img) 
    if key==ord('q'):
        break
    
print(a)
video.release()
cv2.destroyAllWindows()