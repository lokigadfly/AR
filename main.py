import cv2
import numpy as np
from cv2 import imshow






cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
def count_non_zero(y,x):
    global h,w
    counter=0
    # print (h,w)
    for i in range(10):
        for j in range(10):
            if y+i< h and x+j <w and y+i>=0 and x+j>=0:
                if contour[y+i,x+j]>10:
                    counter+=1
    return counter
img=video_capture.read()[1]
img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
global h,w
h,w,_=img.shape
tri_info=[]
rect_info=[]
while True:
    ##################################### Preprocess the image ####################################
    img=video_capture.read()[1]
    img = cv2.resize(img,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
    
    ################################### Find contours of rect and tri ####################################
    _,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    # imshow('contour',img)
    ################################### Locate tri and rect ###################################
    
    for cont in contours:
        if cv2.contourArea(cont)>=300 and cv2.contourArea(cont)<=h*w/4:
            tri=np.array([[0,0],[0,0],[0,0]])
            rect=np.array([[0,0],[0,0],[0,0],[0,0]])
            arc_len = cv2.arcLength(cont,True) 
            approx = cv2.approxPolyDP( cont, 0.03 * arc_len, True )
            if (len(approx)==3):
                
                for i in range(3):
                    tri[i]=approx[i][0]
                edge1=(tri[0][0]-tri[1][0])**2+(tri[0][1]-tri[1][1])**2
                edge2=(tri[0][0]-tri[2][0])**2+(tri[0][1]-tri[2][1])**2
                edge3=(tri[1][0]-tri[2][0])**2+(tri[1][1]-tri[2][1])**2
                # print(edge1/edge2,edge1/edge3,edge2/edge3)    
                # if (edge1/edge2<=1.2 and edge1/edge2>0.8 and edge1/edge3 <= 1.2 and edge1/edge3 > 0.8 and edge2/edge3 <= 1.2 and edge2/edge3 >0.8):             
                if edge1+edge2+edge3/cv2.contourArea(cont)<5000:
                    if (len(tri_info)==0):
                        tri_info.append(tri)
                    img=cv2.polylines(img,[tri],True,(0,255,0),3)
            if (len(approx)==4):
                for i in range(4):
                    rect[i]=approx[i][0]
                edge1=(rect[0][0]-rect[1][0])**2+(rect[0][1]-rect[1][1])**2#    1   4
                edge2=(rect[1][0]-rect[2][0])**2+(rect[1][1]-rect[2][1])**2#    2   3   
                edge3=(rect[2][0]-rect[3][0])**2+(rect[2][1]-rect[3][1])**2
                edge4=(rect[3][0]-rect[0][0])**2+(rect[3][1]-rect[0][1])**2
                diagonal1=(rect[0][0]-rect[2][0])**2+(rect[0][1]-rect[2][1])**2
                diagonal2=(rect[1][0]-rect[3][0])**2+(rect[1][1]-rect[3][1])**2
                # print(edge1+edge2+edge3+edge4/cv2.contourArea(cont))
                if edge1+edge2+edge3+edge4/cv2.contourArea(cont)<6000:
                # if ( edge1/edge2<1.2  and edge1/edge3<1.2 and edge1/edge4<1.2 and diagonal1/diagonal2<1.2 and diagonal1/diagonal2>0.8):
                    if (len(rect_info)<4):
                        rect_info.append(rect)
                    img=cv2.polylines(img,[rect],True,(0,0,255),3)
                # img=cv2.polylines(img,[rect],True,(0,0,255),3)
    
    if (len(rect_info)<4):
        rect_info=[]
    elif (len(rect_info)==4):
        cv2.putText(img, 'Found 4 rect', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
    imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
all_info=rect_info+tri_info
all_center=[]

for fig in all_info:
    center = [sum([fig[i][0] for i in range(len(fig))])/len(fig),sum([fig[i][1] for i in range(len(fig))])/len(fig)]
    all_center.append(center)
# print(all_info)
# print(all_center)
while True:
    img=video_capture.read()[1]
    img = cv2.resize(img,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
    _,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.putText(img, 'Choosing a fig.', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
    #print the located graphs on screen
    if len(tri_info)!=0:
        for i in range(len(tri_info)):
            img=cv2.polylines(img,[tri_info[i]],True,(0,255,0),3)
    if len(rect_info)!=0:
        for i in range(len(rect_info)):
            img=cv2.polylines(img,[rect_info[i]],True,(0,0,255),3)
    #Find finger
    for cont in contours:
    # print(cv2.contourArea(cont))
        if cv2.contourArea(cont)>=300 and cv2.contourArea(cont)<=h*w/4:
            arc_len = cv2.arcLength(cont,True) 
            approx = cv2.approxPolyDP( cont, 0.03 * arc_len, True )
            if (len(approx)==5):
                penta=np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
                for i in range(5):
                    penta[i]=approx[i][0]
                edge=[]
                edge.append((penta[0][0]-penta[1][0])**2+(penta[0][1]-penta[1][1])**2)
                edge.append((penta[1][0]-penta[2][0])**2+(penta[1][1]-penta[2][1])**2)
                edge.append((penta[2][0]-penta[3][0])**2+(penta[2][1]-penta[3][1])**2)
                # edge.append((penta[3][0]-penta[4][0])**2+(penta[3][1]-penta[4][1])**2)
                # edge.append((penta[4][0]-penta[0][0])**2+(penta[4][1]-penta[0][1])**2)
                if (max(edge)/min(edge)<2.3 and max(edge)/min(edge)>1.7):
                    img=cv2.polylines(img,[penta],True,(0,100,100),3)
                    # img=cv2.circle(img,(penta[0][0],penta[0][1]),3,(255,255,255),3)
                    # img=cv2.circle(img,(penta[1][0],penta[1][1]),5,(255,255,255),3)
                    # img=cv2.circle(img,(penta[2][0],penta[2][1]),7,(255,255,255),3)
                    # img=cv2.circle(img,(penta[3][0],penta[3][1]),9,(255,255,255),3)
                    # img=cv2.circle(img,(penta[4][0],penta[4][1]),11,(255,255,255),3)
                    distance=[]
                    for i in range(len(all_center)):
                        center=all_center[i]
                        # distance.append(np.sqrt((center[0]-penta[0][0])**2+(center[1]-penta[0][1])**2))
                        if (np.sqrt((center[0]-penta[0][0])**2+(center[1]-penta[0][1])**2)<50):
                            cv2.putText(img, 'This one?', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
                            img=cv2.polylines(img,[all_info[i]],True,(255,255,255),3)
                            locked_fig=all_info[i]
                            center[0]=int(center[0])
                            center[1]=int(center[1])
                            original_center = np.array(center)
                            break
    imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('l'):
        break

while True:
    img=video_capture.read()[1]
    img = cv2.resize(img,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
    _,contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.putText(img, 'Successfully locking a fig.', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
    img=cv2.polylines(img,[locked_fig],True,(255,255,255),3)
    for cont in contours:
        if cv2.contourArea(cont)>=300 and cv2.contourArea(cont)<=h*w/4:
            arc_len = cv2.arcLength(cont,True) 
            approx = cv2.approxPolyDP( cont, 0.03 * arc_len, True )
            if (len(approx)==5):
                penta=np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
                for i in range(5):
                    penta[i]=approx[i][0]
                edge=[]
                edge.append((penta[0][0]-penta[1][0])**2+(penta[0][1]-penta[1][1])**2)
                edge.append((penta[1][0]-penta[2][0])**2+(penta[1][1]-penta[2][1])**2)
                edge.append((penta[2][0]-penta[3][0])**2+(penta[2][1]-penta[3][1])**2)
                # edge.append((penta[3][0]-penta[4][0])**2+(penta[3][1]-penta[4][1])**2)
                # edge.append((penta[4][0]-penta[0][0])**2+(penta[4][1]-penta[0][1])**2)
                if (max(edge)/min(edge)<2.3 and max(edge)/min(edge)>1.7):
                    img=cv2.polylines(img,[penta],True,(0,100,100),3)
                    center=np.array([int(penta[0][0]),int(penta[0][1])])
                    vect=center-original_center
                    # print('c,oc,v',center,original_center,vect)
                    new_fig=[]
                    for i in range(len(locked_fig)):
                        new_fig.append(locked_fig[i]+vect)
                    new_fig=np.array(new_fig)
                    for i in range(len(locked_fig)):
                        new_fig[i][0]=int(new_fig[i][0])
                        new_fig[i][1]=int(new_fig[i][1])
                    img=cv2.polylines(img,[new_fig],True,(255,255,255),3)
                    for i in range(len(locked_fig)):
                        print(locked_fig[i],new_fig[i])
                        img=cv2.polylines(img,[np.array([locked_fig[i],new_fig[i]])],True,(255,255,255),3)
    imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()