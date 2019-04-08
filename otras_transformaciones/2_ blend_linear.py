import cv2

src1 = cv2.imread(cv2.samples.findFile('../resources/beach.jpg'))
src2 = cv2.imread(cv2.samples.findFile('../resources/family.jpg'))

alpha = 0.8
beta = (1.0 - alpha)
dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

cv2.imshow('Blend Linear', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()
