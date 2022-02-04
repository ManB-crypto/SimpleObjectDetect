import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco2.names', 'r') as f:
    classes = f.read().splitlines()

capture = cv2.VideoCapture('video.mp4')
capture.set(3, 640)
capture.set(4, 480)

while True:
    _, img = capture.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class1 = []

    for outputs in layerOutputs:
        for detection in outputs:
            scores = detection[5:]
            class2 = np.argmax(scores)
            confidence = scores[class2]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class1.append(class2)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes.flatten())


    size=(len(boxes), 3)
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class1[i]])
        confidence = str(round(confidences[i], 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(160, 32, 240))
        cv2.putText(img, label + " " + confidence, (x, y + 20),cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(2)


capture.release()
cv2.destroyAllWindows()
