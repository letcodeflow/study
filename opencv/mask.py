import cv2
import sys

src = cv2.imread('airplane.bmp', cv2.IMREAD_COLOR)
mask  = cv2.imread('mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('field.bmp', cv2.IMREAD_COLOR)

cv2.copyTo(src,mask, dst)

dst[mask > 0] = src[mask > 0]

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()

cv2.destroyAllWindows()
