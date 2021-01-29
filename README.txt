RUN:
python people_detection.py --input GOPR5826.mp4 --output output/out.mp4 --yolo yolo-coco --confidence 0.3 --threshold 0.5

The output video is created inside a directory called 'output'.

EXTERNAL RESOURCES:
'yolo-coco' folder contains coco.names, yolov3.cfg and the link for the yolov3.weights download.
'localization' folder contains the images for template matching in localization.py
'data' folder contains haarcascade_frontalface_default.xml for face detection.

VIDEOs FOR TESTING:
VIRB0399
GOPR5826
GOPR5829
IMG_7845
IMG_7850
IMG_9624
IMG_7857
VIRB0415
VIRB0420
VID_20180529_112800
VID_20180529_112951
IMG_4075
IMG_4084
VID_20180529_113001


people_detection.py:
- pipeline start
- paintings_detection function call
- people and faces detection
- localization function call
- output video creation

paintings_detection.py:
- paintings detections
- paintings rectification
- paintings_retrieval function call

paintings_retrieval.py:
- paintings retrieval 

localization.py:
- template matching for visual localization