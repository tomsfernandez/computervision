import cv2 as cv
import argparse
import numpy as np

from common import *
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

backends = (
    cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
    cv.dnn.DNN_BACKEND_OPENCV)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD)
modelFilePath = "yolo.weights"
configFilePath = "yolo.cfg"
classesFilePath = "object_detection_classes_pascal_voc.txt"
framework = "darknet"
default_backend = cv.dnn.DNN_BACKEND_DEFAULT
default_target = cv.dnn.DNN_TARGET_CPU

model = findFile(modelFilePath)
config = findFile(configFilePath)
classes = findFile(classesFilePath)

# If config specified, try to load it as TensorFlow Object Detection API's pipeline.
config = readTextMessage(config)
if 'model' in config:
    print('TensorFlow Object Detection API config detected')
    if 'ssd' in config['model'][0]:
        print('Preparing text graph representation for SSD model: ' + args.out_tf_graph)
        createSSDGraph(model, config, args.out_tf_graph)
        config = args.out_tf_graph
    elif 'faster_rcnn' in config['model'][0]:
        print('Preparing text graph representation for Faster-RCNN model: ' + args.out_tf_graph)
        createFasterRCNNGraph(model, config, args.out_tf_graph)
        config = args.out_tf_graph

# Load names of classes
classes = None
if classes:
    with open(classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

# Load a network
net = cv.dnn.readNetFromDarknet(modelFilePath)
net.setPreferableBackend(default_backend)
net.setPreferableTarget(default_target)
outNames = net.getUnconnectedOutLayersNames()

confThreshold = args.thr
nmsThreshold = args.nms


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert (classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
                     cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width * height <= 1:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)


def callback(pos):
    global confThreshold
    confThreshold = pos / 100.0


cv.createTrackbar('Confidence threshold, %', winName, int(confThreshold * 100), 99, callback)

cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0)
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Create a 4D blob from a frame.
    inpWidth = args.width if args.width else frameWidth
    inpHeight = args.height if args.height else frameHeight
    blob = cv.dnn.blobFromImage(frame, args.scale, (inpWidth, inpHeight), args.mean, args.rgb, crop=False)

    # Run a model
    net.setInput(blob)
    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        frame = cv.resize(frame, (inpWidth, inpHeight))
        net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')
    outs = net.forward(outNames)

    postprocess(frame, outs)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv.imshow(winName, frame)
