import cv2

face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("photo.jpg",0)
img_resize= cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))



face= face_cascade.detectMultiScale(img_resize, scaleFactor=1.05, minNeighbors=10)
for x,y,w,h in face:
    img=cv2.rectangle(img_resize,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow("Naman" , img)
cv2.waitKey(0)
cv2.destroyAllWindows()