import cv2

img = cv2.imread('img.jpeg', 0)
Gaussian = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
