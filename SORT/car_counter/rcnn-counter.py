import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision
import torch
import argparse
from PIL import Image
import cvzone
from sort import *
import math

cap = cv2.VideoCapture("../data/video/vid1.mp4")
cap.set(3,1280)
cap.set(4,720)

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold,detections):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image
    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    for i in range(len(pred_classes)):
    	if(pred_classes[i] == "car"):
    		currentArray = np.array([pred_bboxes[i][0],pred_bboxes[i][1],pred_bboxes[i][2],pred_bboxes[i][3],int(pred_scores[i])])
    		detections = np.vstack((detections,currentArray))
    return boxes, pred_classes, outputs[0]['labels'], detections


def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        if(classes[i]=='car'):
            color = COLORS[labels[i]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                        lineType=cv2.LINE_AA)
    return image



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits = [103,200,673,200]
totalCount = []

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

while True:
	success, img = cap.read()
	detections = np.empty((0,5))
	boxes, classes, labels, detections = predict(img, model, device, 0.3,detections)
	img = draw_boxes(boxes, classes, labels, img)

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

	cvzone.putTextRect(img,f'Count : {len(totalCount)}',(50,50),scale=1,thickness=1,offset=3)	
	img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
	cv2.imshow("Image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'): break

