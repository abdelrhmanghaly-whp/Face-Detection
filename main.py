import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imgs'
images = []
Names = []
myList = os.listdir(path)
print(myList)

for i in myList:
    currImg = cv2.imread(f'{path}/{i}')
    images.append(currImg)
    Names.append(os.path.splitext(i)[0])

print(Names)

def findEncodings(imgs):
    encodedList = []
    for i in imgs:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        try:
            encoded = face_recognition.face_encodings(i)[0]
            encodedList.append(encoded)
        except IndexError:
            print(f"No face found in image: {i}")

    return encodedList

def markAttend(name):
    with open('attendance.csv', 'r+') as ar:
        dataList = ar.readlines()
        namesList = []
        for i in dataList:
            entry = i.split(',')
            namesList.append(entry[0])
        if name not in namesList:
            curr = datetime.now()
            converted_dt_str = curr.strftime('%H:%M:%S')
            ar.writelines(f'\n{name},{converted_dt_str}')

known = findEncodings(images)

if not known:
    print("No faces found in the provided images. Exiting.")
    exit()

print("Encoding Completed!")

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    resized = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    frames = face_recognition.face_locations(resized)
    currEncoding = face_recognition.face_encodings(resized, frames)

    for currFace, loc in zip(currEncoding, frames):
        matches = face_recognition.compare_faces(known, currFace)
        dist = face_recognition.face_distance(known, currFace)
        idx = np.argmin(dist)

        if matches[idx]:
            name = Names[idx].upper()
            y1, x2, y2, x1 = loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttend(name)

    cv2.imshow('Camera', img)
    if cv2.waitKey(1) == 27:
        break
