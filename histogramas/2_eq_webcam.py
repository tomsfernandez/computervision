import cv2

print(f"Ejercicio 2: Ecualizar la webcam monocromática en tiempo real")
print(f"With OpenCV version: {cv2.__version__}")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    equalized_frame = cv2.equalizeHist(grey_frame)
    cv2.imshow('Source image', grey_frame)
    cv2.imshow('Equalized Image', equalized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
