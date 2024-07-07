import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import logging

logging.basicConfig(filename='face_recognition.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

path = 'imgs'
images = []
Names = []

try:
    myList = os.listdir(path)
    logging.info(f"Found files: {myList}")
except FileNotFoundError:
    logging.error(f"Directory '{path}' not found.")
    exit()
except Exception as e:
    logging.error(f"Error reading directory '{path}': {e}")
    exit()

for i in myList:
    try:
        currImg = cv2.imread(f'{path}/{i}')
        if currImg is None:
            logging.warning(f"Image '{i}' not found or unable to read.")
            continue
        images.append(currImg)
        Names.append(os.path.splitext(i)[0])
    except Exception as e:
        logging.error(f"Error loading image '{i}': {e}")

logging.info(f"Loaded names: {Names}")

def findEncodings(imgs):
    encodedList = []
    for i in imgs:
        try:
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            encoded = face_recognition.face_encodings(i)[0]
            encodedList.append(encoded)
        except IndexError:
            logging.warning("No face found in image.")
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
    return encodedList

def markAttend(name):
    try:
        if not os.path.exists('attendance.csv'):
            with open('attendance.csv', 'w') as ar:
                ar.write('name,time\n')
        with open('attendance.csv', 'r+') as ar:
            dataList = ar.readlines()
            namesList = [entry.split(',')[0] for entry in dataList]
            if name not in namesList:
                curr = datetime.now()
                converted_dt_str = curr.strftime('%H:%M:%S')
                ar.write(f'{name},{converted_dt_str}\n')
    except Exception as e:
        logging.error(f"Error updating attendance: {e}")

known = findEncodings(images)

if not known:
    logging.error("No faces found in the provided images. Exiting.")
    exit()

logging.info("Encoding Completed!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Error: Camera not accessible.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        logging.error("Failed to capture image from camera.")
        break
    
    try:
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
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttend(name)

        cv2.imshow('Camera', img)
        if cv2.waitKey(1) == 27:
            break
    except Exception as e:
        logging.error(f"Error during face recognition: {e}")

cap.release()
cv2.destroyAllWindows()
