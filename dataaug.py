import cv2
import numpy as np
img = cv2.imread('./train/1/7-3-2864-16.jpg',0)
# img = cv2.resize(img,(200,600))

img2 = img.copy()
for i in range(4):
    x = np.random.randint(0, 63)
    # y = np.random.randint(0, 8)
    xoft = np.random.randint(0, 63 - x)
    # yoft = np.random.randint(0, 9 - y)

    b = np.random.randint(20, 40)

    img2[x:x + xoft, :] += b

img = cv2.resize(img,(200,600))
img2 = cv2.resize(img2,(200,600))
# cv2.imshow('1',img)
cv2.imshow('2',img2)
cv2.waitKey(0)
