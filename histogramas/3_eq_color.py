import cv2

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    cv2.imshow('Color input image', frame)
    cv2.imshow('Histogram equalized', img_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
