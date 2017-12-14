import  cv2
import numpy as np



#载入并显示图片
img=cv2.imread('2.JPG')
img = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
cv2.imshow('img',img)

#灰度化
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
contour = gray
#输出图像大小，方便根据图像大小调节minRadius和maxRadius
#霍夫变换圆检测
circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,0.1,param1=20,param2=50,minRadius=100,maxRadius=200)
#输出返回值，方便查看类型
#输出检测到圆的个数
print(len(circles[0]))

# Find best match circle
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
contour = cv2.filter2D(contour,-1,kernel)

def count_non_zero(y,x):
	counter=0
	for i in range(10):
		for j in range(10):
			if contour[y+i,x+j]!=0:
				counter+=1
	return counter
cv2.imshow('contour',contour)
# print('shape',contour.shape)
# print('contour: ',len(contour))
#根据检测到圆的信息，画出每一个圆
# print(contour[0:3,0:3])
mean=[]
max_sum=[]
for circle in circles[0]:
    #坐标行列
    x=int(circle[0])
    y=int(circle[1])
    #半径
    r=int(circle[2])
    up_average = count_non_zero(y+r-5,x-5)
    down_average = count_non_zero(y-r-5,x-5)
    right_average = count_non_zero(y-5,x+r-5)
    left_average = count_non_zero(y-5,x-r-5)
    print(up_average,down_average,right_average,left_average)
    # img=cv2.circle(img,(x,y),r,(0,0,255),1)		
    # mean.append(up_average+down_average+left_average+right_average)
    if (up_average>0 and down_average>0 and right_average>0 and left_average>0):
    	#在原图用指定颜色标记出圆的位置
    	mean.append([x,y,r])
    	max_sum.append(sum([up_average,down_average,right_average,left_average]))
#显示新图像
for i in range(len(mean)):
	if (max_sum[i] == max(max_sum)):
		img=cv2.circle(img,(mean[i][0],mean[i][1]),mean[i][2],(0,0,255),1)		
cv2.imshow('res',img)

#按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()