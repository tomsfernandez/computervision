import cv2 as cv

src1 = cv.imread(cv.samples.findFile('../resources/beach.jpg'))
src2 = cv.imread(cv.samples.findFile('../resources/family.jpg'))

alpha = 0.5
beta = (1.0 - alpha)
dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)

cv.imshow('Blend Linear', dst)
cv.waitKey(0)

cv.destroyAllWindows()
