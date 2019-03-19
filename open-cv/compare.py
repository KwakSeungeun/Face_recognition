import numpy as np
import cv2

# openface에서 제공하는 classifier사용
face_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')

image = cv2.imread("../images/testImage3.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.03, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y),(x+w, y+h),(0,255,0),2)


# 실행 창 띄우기
cv2.imshow("face detection test", image) 
cv2.waitKey(0)
# 종료
cv2.destoryAllWindows()

# Classifier가 정확하지 않음 => 얼굴이 아닌 부분도 인식하는 경우가 있음
# Feature matching을 사용하여 찾고자 하는 사람의 얼굴이 들어있는 사진을 모을 수 있을 것으로 생각.