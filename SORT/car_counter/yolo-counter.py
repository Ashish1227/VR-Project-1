from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("../data/video/vid2.mp4")
cap.set(3,1280)
cap.set(4,720)

model = YOLO('../yolo-weights/yolov8n.pt')

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant','stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe','backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle','wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed','dining table',
    'toilet','tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator','book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [200,400,1100,400]
totalCount = []


while True:

	success, img = cap.read()

	results = model(img,stream=True)

	detections = np.empty((0,5))

	for r in results:
		boxes = r.boxes
		for box in boxes:
			x1,y1,x2,y2 = box.xyxy[0]
			x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
			w,h = x2-x1,y2-y1
			bbox = int(x1),int(y1),int(w),int(h)
			#cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
			
			conf = box.conf[0]
			conf = math.ceil((box.conf[0]*100))/100
			print(conf)
			#class name
			cls = int(box.cls[0])
			currentClass = classNames[cls]

			if currentClass == "car" and conf > 0.3:
				# cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=3)
				# cvzone.cornerRect(img,bbox,l=4,rt=2)
				currentArray = np.array([x1,y1,x2,y2,conf])
				detections = np.vstack((detections,currentArray))


	resultsTracker = tracker.update(detections)
	cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

	for result in resultsTracker:
		x1,y1,x2,y2,Id = result
		x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
		print(result)
		w,h = x2-x1,y2-y1
		bbox = int(x1),int(y1),int(w),int(h)
		cvzone.cornerRect(img,bbox,l=9,rt=5,colorR=(255,0,0))
		cvzone.putTextRect(img,f'{int(Id)}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
	
		cx,cy = x1+w//2,y1+h//2
		cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

		if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
			if totalCount.count(Id) == 0:
				totalCount.append(Id)

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	codec = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('./outputs/vid1.avi', codec, fps, (width, height))
	cvzone.putTextRect(img,f'Count : {len(totalCount)}',(50,50),scale=1,thickness=1,offset=3)
	out.write(img)	
	cv2.imshow("Image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'): break