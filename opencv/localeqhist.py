import cv2

img = cv2.imread('img.jpeg', 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv2.imwrite('localhist.jpg',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()