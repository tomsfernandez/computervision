import cv2

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        src = param.copy()

        # Hay que refinar los parametros
        cv2.floodFill(src, None, (x, y), (0, 255, 255), 3, 3, flags)
        cv2.imshow('Fill zone', src)


while True:
    ret, frame = video_capture.read()

    cv2.setMouseCallback('Source Image', on_mouse, frame)
    cv2.imshow('Source Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

