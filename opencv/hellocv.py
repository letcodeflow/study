import sys
import cv2

print('fucking cv', cv2.__version__)

img = cv2.imread('cat.bmp')

if img is None:
    print('there is no img')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()

cv2.destroyAllWindows()