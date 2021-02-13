import cv2

scale = 1
delta = 0
depth = cv2.CV_16S

src = cv2.imread('img.jpeg', 0)
src = cv2.GaussianBlur(src, (3, 3), 0)

grad_x = cv2.Sobel(src, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(src, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow('Sobel', grad)
cv2.waitKey(0)
cv2.destroyAllWindows()