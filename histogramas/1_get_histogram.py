import cv2
import numpy as np
from builtins import enumerate
from matplotlib import pyplot as plt

print(f"Calcular el histograma monocromático y color de la webcam en tiempo real")
print(f"With OpenCV version: {cv2.__version__}")

color = ('b','g','r')
video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()

def plot_gray_scale_hist(image):
    plt.figure(1)
    plt.clf()
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.hist(grey_image.ravel(), 256, [0, 256])
    plt.xlim([0,256])
    plt.ylim([0, 5000])

def plot_color_scale_hist(image):
    color = ('b','g','r')
    plt.figure(2)
    plt.clf()
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        plt.ylim([0, 5000])

while True:
    ret, frame = video_capture.read()
    plot_gray_scale_hist(frame)
    plot_color_scale_hist(frame)
    cv2.imshow('Source image', frame)
    plt.draw()
    plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.close('all')