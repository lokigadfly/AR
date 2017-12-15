import cv2
import numpy as np

CANNY=250
MORPH = 11
img=cv2.imread('1.JPG')
img = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
cv2.imshow('img',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
contour = gray

gray = cv2.bilateralFilter(gray,2,10,120)
# ret,thresh = cv2.threshold(gray,127,255,0)
edges = cv2.Canny(gray,10,CANNY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )
image,contours,hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
# img = cv2.drawContours(image, contours, -1, (0,255,0), 3)
for cont in contours:
	# print(cv2.contourArea(cont))
	if (cv2.contourArea(cont)>=100):
		arc_len = cv2.arcLength(cont,True)
		approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )
		if (len(approx)==3):
			print(1)
			#改len(approx)==3 or 4 判断是三角形还是四角形



# cv2.imshow('gray',)



cv2.waitKey(0)
cv2.destroyAllWindows()
