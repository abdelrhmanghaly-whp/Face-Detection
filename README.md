# Face Detection 

This project implements a simple face detection system using Python, OpenCV, and face_recognition libraries. The script captures frames from the webcam, detects faces, compares them with pre-encoded faces, and marks attendance if a recognized face is detected.

## Prerequisites
Make sure you have the following libraries installed:

```bash
pip install opencv-python
pip install numpy
pip install face_recognition
```
## Getting Started
* Clone the repository:
  ```bash
  git clone https://github.com/your-username/face-detection-project.git
  cd face-detection-project
  ```
* Place The images in the imgs directory
* Run the script using:
  ```bash
  python main.py
  ```
 ## Code Explaination
  ### Libraries Used
* OpenCV (cv2): Used for capturing video frames and image processing.
* NumPy (numpy): Used for numerical operations on arrays.
* face_recognition: A face recognition library that simplifies face recognition using pre-trained models.
 ### Code Overview
   ### Loading Images:

     * Images for recognition are loaded from the imgs directory.
     * Image names are used as labels.
   ### Encoding Faces:

     * findEncodings function converts BGR images to RGB and encodes faces using face_recognition.face_encodings.
     * Encoded faces are stored in the known variable.
   ### Marking Attendance:

     * markAttend function appends the name and current time to the attendance.csv file if the face is not already marked.
   ### Face Recognition Loop:

     * The script continuously captures frames from the webcam.
     * Detected faces are compared with the pre-encoded faces.
     * If a match is found, the person's name is displayed, and attendance is marked.
   ### Display:

     * The script displays the video feed with rectangles around detected faces and names.

  
