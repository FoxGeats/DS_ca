import cv2
import numpy as np
import face_recognition
import os
from PIL import Image, ImageDraw

path="faceImages"
images=[]
classNames=[]
myList = os.listdir(path)


for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # get name only




def findEncodings(images):
    encodeList =[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



encodeListKnown= findEncodings(images)
print("encoding complete!")

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS= cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesInCurrentFrame = face_recognition.face_locations(imgS) #multiple faces
    encodingsCurrentFrame=face_recognition.face_encodings(imgS,facesInCurrentFrame)

    for encodeFace,faceLocation in zip(encodingsCurrentFrame,facesInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLocation
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(img)

    pil_image = Image.fromarray(img)
    for face_landmarks in face_landmarks_list:
      d = ImageDraw.Draw(pil_image, 'RGBA')

       # Make the eyebrows into a nightmare
      d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
      d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
      d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
      d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

      # Gloss the lips
      d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
      d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
      d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
      d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

      # Sparkle the eyes
      d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
      d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

      # Apply some eyeliner
      d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
      d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)


    img2 = np.array(pil_image)
    cv2.imshow("Webcam",img2)
    cv2.waitKey(1)











