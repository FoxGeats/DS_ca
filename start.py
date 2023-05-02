import cv2
import numpy as np
import face_recognition

imgJC= face_recognition.load_image_file("faceImages/JC.jpg")
imgJC = cv2.cvtColor(imgJC,cv2.COLOR_BGR2RGB)

imgJCTest= face_recognition.load_image_file("faceImages/John_Cena.jpg")
imgJCTest =cv2.cvtColor(imgJCTest,cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgJC)[0]
encodeKR=face_recognition.face_encodings(imgJC)[0]
cv2.rectangle(imgJC,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(0,255,0),2)

faceLocationTest = face_recognition.face_locations(imgJCTest)[0]
encodeKRTest=face_recognition.face_encodings(imgJCTest)[0]
cv2.rectangle(imgJCTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(0,255,0),2)

results= face_recognition.compare_faces([encodeKR],encodeKRTest) 

faceDist= face_recognition.face_distance([encodeKR],encodeKRTest)
print(results,faceDist)

cv2.putText(imgJCTest,f'{results} {round(faceDist[0],2)}',(50,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

cv2.imshow('John Cena',imgJC)
cv2.imshow('John Cena Test',imgJCTest)

cv2.waitKey(0)







