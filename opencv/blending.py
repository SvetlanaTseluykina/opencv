import cv2

foreground = cv2.imread("puppets.png")
background = cv2.imread("coast1.jpg")
alpha = cv2.imread("puppet_alpha.png")

foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float) / 255

# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)

outImage = cv2.add(foreground, background)

cv2.imshow("outImg", outImage / 255)
cv2.waitKey(0)