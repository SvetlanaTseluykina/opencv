import cv2

scale = 1
delta = 0
depth = cv2.CV_16S

src = cv2.imread('img.jpeg', 0)
src = cv2.GaussianBlur(src, (3, 3), 0)

lpls = cv2.Laplacian(src, depth, ksize=3)
abs_grad = cv2.convertScaleAbs(lpls)
cv2.imshow('Laplacian', abs_grad)
cv2.waitKey(0)
cv2.destroyAllWindows()