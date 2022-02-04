# SimpleObjectDetect
Simpler version of Object Detection
### Requirements
* coco.names " is a data set that help the program identify the objects"
* yolov3.cfg " is a algorithm that identifies specific objects in videos in real time"
* yolov3.weights (*have to download urself*) " it is the weights of the binary file"

### Features
* Uses OpenCV and NumPY libary.
* Able to identfy 80 objects with high accuracy.
* YOLOv3 has the advantages of detection speed and accuracy.
* Easy to learn and implement.

### Customizable
* capture = cv2.VideoCapture('') "the video can be change to the name of the video *video.mp4* or neumerical for USB webcam."
* cv2.FONT_HERSHEY_PLAIN "the font type can be change to others type to be more suitable"
* color=(160, 32, 240) "the box and fond color"
*  cv2.imshow('Image', img) " the name of the window"
*  if confidence > 0.5: " the confidence level"
### Example
![image](https://user-images.githubusercontent.com/80488842/152525672-e23b5a6b-5dcc-4b03-9aa7-e2223d483ca2.png)
