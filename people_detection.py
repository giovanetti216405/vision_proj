import numpy as np
import argparse
import time
import cv2
import os
import paintings_detection as pd
import localization as loc

# ============================ Condition to avoid that a painting's subject is detected as 'person' ========================================


def person_inside_painting():
    if paints_data:
        for i in range(len(paints_data)):
            data = paints_data[i]
            (xp, yp) = (data[0], data[1])
            (wp, hp) = (data[2], data[3])

            if x > xp and x + w < xp + wp and y > yp and y + h < yp + hp:
                    return True
        return False
    else:
        return False

# ============================= Resize of the images in the DB and creation of the database list ===================================

imgs_db = []
for img in os.listdir("db_images"):
    p = os.path.join("db_images", img).replace("\\", "/")
    img = cv2.imread(p)
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    imgs_db.append(img)

# ======================Parse parameters command line========================
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialization a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
# load the YOLO object detector
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# load input video
cap = cv2.VideoCapture(args["input"])

# initializations
writer = None
(W, H) = (None, None)
count_frame = 0
frame = None

# =================================== Video reading ========================================
while cap.isOpened():

    print("Frame ", count_frame)
    count_frame += 1

    ret, frame = cap.read()
    if not ret:
        break

    if frame is not None:

        # Frame resize
        frame = cv2.resize(src=frame, dsize=(0, 0), dst=None, fx=0.5, fy=0.5)


        # ======================================PAINTINGS DETECTION====================================================
        # frame = current frame
        # paints_data = information about all the paintings in the current frame
        # room = room of the current frame
        frame, paints_data, room = pd.paintings_detection(imgs_db, frame)

        (H, W) = frame.shape[:2]

        # determine only the output layer names that we need from YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # Three lists initialization: bounding boxes, confidence values and class IDs
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

        # Initialization for Face Detection classifier
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_cascade_face = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # draw the bounding box only around people with a confidence greater than 0.7
                if LABELS[classIDs[i]] == 'person' and confidences[i] > 0.70:
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw the bounding box only if the detected person is not inside a painting (a painting's subject)
                    if not person_inside_painting():

                            if room != 0:
                                print("******THE DETECTED PERSON IS IN THE ROOM ", room)

                            #  ===========================================FACE DETECTION =====================================

                            faces_rects = haar_cascade_face.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5)

                            # check if the detected person is facing a painting
                            if len(faces_rects) != 0:
                                for (xf, yf, wf, hf) in faces_rects:
                                    if xf > x and xf + wf < x + w and yf > y and yf + hf < y + h:
                                        cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 2)
                                        print("******THE DETECTED PERSON IS FACING THE CAMERA, NOT A PAINTING")
                                    else:
                                        print("******THE DETECTED PERSON IS FACING A PAINTING")
                            else:
                                print("******THE DETECTED PERSON IS FACING A PAINTING")

                            # _____________________________________________________________________________________________________

                            # draw bounding box around the detected person
                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #  ===========================================LOCALIZATION =====================================
        # only if at least a painting is detected, otherwise there are not information for localization
        if paints_data:
            loc.localize(room)

        cv2.imshow('frame', frame)

        # ==================================VIDEO WRITER========================================================
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            # write the output frame to disk
        writer.write(frame)

    else:
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
print("[INFO] cleaning up...")
writer.release()
cap.release()
cv2.destroyAllWindows()