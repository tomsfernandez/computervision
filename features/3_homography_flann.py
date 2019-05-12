import numpy as np
import cv2

MIN_MATCH_COUNT = 10

sift = cv2.xfeatures2d.SIFT_create()

video_capture = cv2.VideoCapture(0)
reference = None

# Borrar
# reference = cv2.imread('../resources/wallet.jpeg')
# kp1, des1 = sift.detectAndCompute(reference, None)

detector = cv2.ORB_create()
cv2.namedWindow("Keypoints")

while True:
    ret, frame = video_capture.read()
    if reference is None:
        reference = frame
        kp1, des1 = sift.detectAndCompute(reference, None)

    kp2, des2 = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w, _ = reference.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print
        "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    frame_with_matches = cv2.drawMatches(reference, kp1, frame, kp2, good, None, **draw_params)

    cv2.imshow('Keypoints', frame_with_matches)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
