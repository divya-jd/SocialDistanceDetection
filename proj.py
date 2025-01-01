import numpy as np
import time
import cv2
import math
import imutils # simplifies basic image processing tasks like resizing, rotating, and cropping images.

labelsPath = "/Users/joisedivya/Documents/MY DOCS_JD/PROJECTS-BTECH/social-distance-detector-master (1)/social-distance-detector-master/coco.names"
LABELS = open(labelsPath).read().strip().split("\n") #strip removes trailing whitespaces and split splits the string into a list

np.random.seed(42) #generates random numbers from 0 to 42
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8") #generates random colors for each label

weightsPath = "/Users/joisedivya/Documents/MY DOCS_JD/PROJECTS-BTECH/social-distance-detector-master (1)/social-distance-detector-master/yolov3.weights"
configPath = "/Users/joisedivya/Documents/MY DOCS_JD/PROJECTS-BTECH/social-distance-detector-master (1)/social-distance-detector-master/yolov3.cfg"
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
# YOLO is a deep learning algorithm that can detect objects from images and videos
# YOLOv3 is the latest variant of a popular object detection algorithm YOLO – You Only Look Once.
# YOLOv3 uses a few tricks to improve training and increase performance, including: multi-scale predictions, a better backbone classifier, and more.
# YOLOv3 is the most popular because it is faster than other algorithms and still has a high accuracy.
# YOLOv3 is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images.
# YOLOv3 is the latest variant of a popular object detection algorithm YOLO – You Only Look Once.

print("Loading Machine Learning Model ...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #readNetFromDarknet reads a network model stored in Darknet model files and config files and returns an instance of cv::dnn::Net class.
# readNetFromDarknet reads a network model stored in Darknet model files and config files and returns an instance of cv::dnn::Net class.


print("Starting Camera ...")
cap = cv2.VideoCapture(0)


while(cap.isOpened()): # starts the loop that continues till video capture
    ret, image = cap.read() # reads the video frame by frame and returns a boolean value and the frame itself 
    if not ret: # if the frame is not returned, the loop breaks
        break

    image = imutils.resize(image, width=800) # resizes the image to a width of 800 pixels and keeps the aspect ratio intact 
    (H, W) = image.shape[:2] # returns the height and width of the image 
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False) # blobFromImage creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels. 
    net.setInput(blob) # sets the new input value for the network
    start = time.time() # returns the time in seconds since the epoch
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames()) # runs a forward pass to compute the net output and returns the layer names of unconnected output layers 
    end = time.time() # returns the time in seconds since the epoch 
    print("Prediction time/frame: {:.6f} seconds".format(end - start)) # prints the time taken to predict the frame 

    boxes = [] # creates an empty list for boxes
    confidences = [] # creates an empty list for confidences # confidence is the probability that the predicted object is correct
    classIDs = [] # creates an empty list for classIDs # classID is the predicted class of the object

    for output in layerOutputs: # for loop to iterate through the layerOutputs list # layerOutputs is a list of lists 
        for detection in output: # for loop to iterate through the detection list # detection is a list of 85 elements. The first 4 elements are the bounding box coordinates, the 5th element is the confidence, and the last 80 elements are the class probabilities. 
            scores = detection[5:] # scores is a list of 80 elements. The first 4 elements are the bounding box coordinates, the 5th element is the confidence, and the last 80 elements are the class probabilities. 
            classID = np.argmax(scores)# returns the indices of the maximum values along an axis # classID is the index of the maximum value in the scores list
            confidence = scores[classID] # confidence is the maximum value in the scores list 

            if confidence > 0.1 and classID == 0: # if the confidence is greater than 0.1 and the classID is 0 (person) 
                box = detection[0:4] * np.array([W, H, W, H]) # box is a list of 4 elements. The first 2 elements are the bounding box coordinates, the 3rd element is the width, and the 4th element is the height.
                (centerX, centerY, width, height) = box.astype("int") # centerX is the x-coordinate of the center of the bounding box, centerY is the y-coordinate of the center of the bounding box, width is the width of the bounding box, and height is the height of the bounding box. # astype() is used to cast a numpy array to a specified type.  
                x = int(centerX - (width / 2))  # x is the x-coordinate of the top left corner of the bounding box # centerX is the x-coordinate of the center of the bounding box, width is the width of the bounding box. #   x = centerX - (width / 2)
                y = int(centerY - (height / 2)) # y is the y-coordinate of the top left corner of the bounding box # centerY is the y-coordinate of the center of the bounding box, height is the height of the bounding box. # y = centerY - (height / 2)
                boxes.append([x, y, int(width), int(height)]) # appends the list of bounding box coordinates to the boxes list # boxes is a list of lists 
                confidences.append(float(confidence)) # appends the confidence to the confidences list # confidences is a list of floats # confidence is the probability that the predicted object is correct 
                classIDs.append(classID) # appends the classID to the classIDs list # classIDs is a list of integers # classID is the predicted class of the object 

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) # NMSBoxes performs non maximum suppression given boxes and corresponding scores. # idxs is a list of integers # boxes is a list of lists # confidences is a list of floats   

    if len(idxs) > 0: # if the length of idxs is greater than 0 
        for i in idxs.flatten():    # for loop to iterate through the idxs list # idxs is a list of integers # flatten() returns a copy of the array collapsed into one dimension.  
            for j in idxs.flatten(): # for loop to iterate through the idxs list # idxs is a list of integers # flatten() returns a copy of the array collapsed into one dimension.
                if i < j: # if i is less than j 
                    x_i, y_i, w_i, h_i = boxes[i]   # x_i is the x-coordinate of the top left corner of the bounding box, y_i is the y-coordinate of the top left corner of the bounding box, w_i is the width of the bounding box, and h_i is the height of the bounding box. # boxes is a list of lists # i is an integer
                    x_j, y_j, w_j, h_j = boxes[j]  # x_j is the x-coordinate of the top left corner of the bounding box, y_j is the y-coordinate of the top left corner of the bounding box, w_j is the width of the bounding box, and h_j is the height of the bounding box. # boxes is a list of lists # j is an integer
                    x_dist = abs(x_j - x_i) # x_dist is the absolute value of the difference between x_j and x_i # x_j is the x-coordinate of the top left corner of the bounding box, x_i is the x-coordinate of the top left corner of the bounding box. # x_dist = abs(x_j - x_i)
                    y_dist = abs(y_j - y_i) # y_dist is the absolute value of the difference between y_j and y_i # y_j is the y-coordinate of the top left corner of the bounding box, y_i is the y-coordinate of the top left corner of the bounding box. # y_dist = abs(y_j - y_i)
                    distance = math.sqrt(x_dist * x_dist + y_dist * y_dist) # distance is the square root of the sum of the squares of x_dist and y_dist # x_dist is the absolute value of the difference between x_j and x_i, y_dist is the absolute value of the difference between y_j and y_i. # distance = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                    cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 255, 0), 2) # draws a rectangle on the image # image is the image, (x_i, y_i) is the top left corner of the rectangle, (x_i + w_i, y_i + h_i) is the bottom right corner of the rectangle, (0, 255, 0) is the color of the rectangle, and 2 is the thickness of the rectangle # x_i is the x-coordinate of the top left corner of the bounding box, y_i is the y-coordinate of the top left corner of the bounding box, w_i is the width of the bounding box, and h_i is the height of the bounding box. # cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 255, 0), 2)
                    cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 255, 0), 2) # draws a rectangle on the image # image is the image, (x_i, y_i) is the top left corner of the rectangle, (x_i + w_i, y_i + h_i) is the bottom right corner of the rectangle, (0, 255, 0) is the color of the rectangle, and 2 is the thickness of the rectangle # x_i is the x-coordinate of the top left corner of the bounding box, y_i is the y-coordinate of the top left corner of the bounding box, w_i is the width of the bounding box, and h_i is the height of the bounding box. # cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 255, 0), 2)

                    if distance < 220: # if the distance is less than 220, then the people are too close to each other # 220 is the distance in pixels between two people # distance is the square root of the sum of the squares of x_dist and y_dist # x_dist is the absolute value of the difference between x_j and x_i, y_dist is the absolute value of the difference between y_j and y_i. # distance = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                        # Violation detected
                        cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2) # draws a rectangle on the image # image is the image, (x_i, y_i) is the top left corner of the rectangle, (x_i + w_i, y_i + h_i) is the bottom right corner of the rectangle, (0, 0, 255) is the color of the rectangle, and 2 is the thickness of the rectangle # x_i is the x-coordinate of the top left corner of the bounding box, y_i is the y-coordinate of the top left corner of the bounding box, w_i is the width of the bounding box, and h_i is the height of the bounding box. # cv2.rectangle(image, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2)
                        cv2.rectangle(image, (x_j, y_j), (x_j + w_j, y_j + h_j), (0, 0, 255), 2) # draws a rectangle on the image # image is the image, (x_j, y_j) is the top left corner of the rectangle, (x_j + w_j, y_j + h_j) is the bottom right corner of the rectangle, (0, 0, 255) is the color of the rectangle, and 2 is the thickness of the rectangle # x_j is the x-coordinate of the top left corner of the bounding box, y_j is the y-coordinate of the top left corner of the bounding box, w_j is the width of the bounding box, and h_j is the height of the bounding box. # cv2.rectangle(image, (x_j, y_j), (x_j + w_j, y_j + h_j), (0, 0, 255), 2)

    cv2.imshow("Social Distancing Detector", image) # displays the image # "Social Distancing Detector" is the name of the window, image is the image # cv2.imshow("Social Distancing Detector", image) 
    if cv2.waitKey(1) & 0xFF == ord('q'): # if the key pressed is q, the loop breaks # cv2.waitKey(1) returns the ASCII value of the key pressed # ord('q') returns the ASCII value of q # if cv2.waitKey(1) & 0xFF == ord('q'): 
        break # breaks the loop

cap.release() # releases the video capture
cv2.destroyAllWindows() # destroys all the windows