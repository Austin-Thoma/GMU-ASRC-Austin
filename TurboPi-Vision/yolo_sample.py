# YOLO object detection
import cv2 as cv
import numpy as np
import time

# Load the image
img = cv.imread('TurboPi-Vision/images/robopov.jpg')
cv.imshow('window', img)
cv.waitKey(1)

# Load names of classes and get random colors
classes = open('TurboPi-Vision/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Find the class ID for "sports ball"
sports_ball_class_id = classes.index("sports ball") if "sports ball" in classes else -1
if sports_ball_class_id == -1:
    raise ValueError("Class 'sports ball' not found in 'coco.names'.")

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Determine the output layer
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the blob as input to the network
net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()
print('Forward propagation time:', t - t0)

# Initialize lists to hold the detection information
boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

# Process the output from the network
for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        
        # Only consider "sports ball" detections
        if classID == sports_ball_class_id and confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply non-maxima suppression to avoid overlapping boxes
indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        
        # Draw the bounding box
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Prepare the label with class name and confidence
        label = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        
        # Calculate text size for the label background
        (text_width, text_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_y = max(y, text_height + 10)
        
        # Draw a filled rectangle for the label background
        cv.rectangle(img, (x, y), (x + text_width, y - text_height - 10), color, -1)
        
        # Put the label text on top of the filled rectangle
        cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the final image with detected "sports ball" objects and labels
cv.imshow('window', img)
cv.waitKey(0)
cv.destroyAllWindows()