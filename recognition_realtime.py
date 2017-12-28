import cv2
import numpy as np
from cv2 import imshow
from numpy import *
def perspect(R,T,matx,x,y,z):
    node = mat([x,y,z]).T
    # R=-R
    # T=-T
    Rx = mat([[1,0,0],[0,math.cos(R[0][0]),-math.sin(R[0][0])],[0,math.sin(R[0][0]),math.cos(R[0][0])]])
    Ry = mat([[math.cos(R[1][0]),0,math.sin(R[1][0])],[0,1,0],[-math.sin(R[1][0]),0,math.cos(R[1][0])]])
    Rz = mat([[math.cos(R[2][0]),-math.sin(R[2][0]),0],[math.sin(R[2][0]),math.cos(R[2][0]),0],[0,0,1]])
    newnode = Rx * Ry  * Rz *node
    #print "newnode1",newnode
    newnode =  newnode + T
    #print "newnode2",newnode,newnode[0][0][0],newnode[1][0],newnode[2][0],"x,y",x,y,"R",R,"T",T
    site = mat([newnode[0,0],newnode[1,0],newnode[2,0]]).T
    nsite = matx * site
    # print "perspect x y",nsite#int(nsite[0,0]/nsite[2,0]),int(nsite[1,0]/nsite[2,0])
    if nsite[0,0]/nsite[2,0]!=np.nan and nsite[1,0]/nsite[2,0]!=np.nan :
        return int(nsite[0,0]/nsite[2,0]),int(nsite[1,0]/nsite[2,0])
    else:
        return 0,0
def f_h(R,T,K,p,c): 
    Rx = mat([[1,0,0],[0,math.cos(R[0][0]),-math.sin(R[0][0])],[0,math.sin(R[0][0]),math.cos(R[0][0])]])
    Ry = mat([[math.cos(R[1][0]),0,math.sin(R[1][0])],[0,1,0],[-math.sin(R[1][0]),0,math.cos(R[1][0])]])
    Rz = mat([[math.cos(R[2][0]),-math.sin(R[2][0]),0],[math.sin(R[2][0]),math.cos(R[2][0]),0],[0,0,1]])
    Rxyz = (Rx * Ry * Rz)
    # print "R,T,K,p,c,Rxyz",R,T,K,p,c,Rxyz
    Rxyz.I
    mat(K).I
    mat([p[0],p[1],1]).T
    ori = Rxyz.I * mat(K).I * mat([p[0],p[1],1]).T
    # print "ori",ori,"Rxyz.I",Rxyz.I,"mat(K).I ",mat(K).I ,"mat([p[0],p[1],1]).T",mat([p[0],p[1],1]).T
    T2=[0,0,0]
    T2[0]=T[0][0];T2[1]=T[1][0];T2[2]=T[2][0]
    tran = Rxyz.I * mat(T2).T
    # print "tran", tran,"mat(T2).T",mat(T2).T
    x0 = ori[0,0]
    y0 = ori[1,0]
    z0 = ori[2,0]
    xp = tran[0,0]
    yp = tran[1,0]
    zp = tran[2,0]
    # print "x0,y0,z0,xp,yp,zp",x0,y0,z0,xp,yp,zp
    y = ((c[0]*y0**2/x0)+c[1]+xp*y0-x0*yp)/(y0**2/x0+x0)
    # x = (-y0/x0)*(y-c[1])+c[0]
    # print "y",y     
    z = z0*(y+yp)/y0-zp
    # print "z",z 
    return z
    


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

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(1)    
img=video_capture.read()[1]

# img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
global h,w
h,w,_=img.shape
# print(h,w)
objectpoints=[array([[0.,0.,0.],[0.,480.,0.],[480.,480.,0.],[480.,0.,0.]],float32)]
imagepoints=[array([[[0.,0.]],[[0.,0.]],[[0.,0.]],[[0.,0.]]],float32)]
objectpointsin2d=[[0.,0.],[0.,480.],[480.,480.],[480.,0.]]
perspectivepoints=[[0.,0.],[0.,0.],[0.,0.],[0.,0.]]
count6=0
sixset=np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

tri_info=[]
rect_info=[]
while True:
    ##################################### Preprocess the image ####################################
    img=video_capture.read()[1]
    imgtopers=video_capture.read()[1]
    # img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
    # imshow('s',thresh)
    contour = gray
    ################################### Find contours of rect and tri ####################################
    image,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # img = cv2.drawContours(image, contours, -1, (0,255,0), 3)
    ################################### calibration ###################################
    
    for cont in contours:
        
        # print(cv2.contourArea(cont))
        #print  cv2.contourArea(cont)>=200 and cv2.contourArea(cont)<=h*w/2
        if cv2.contourArea(cont)>=150 and cv2.contourArea(cont)<=h*w/2 :
            arc_len = cv2.arcLength(cont,True) 
            approx = cv2.approxPolyDP( cont, 0.03 * arc_len, True )
            ################################### Locate six ###################################
            if (len(approx)==6):#first
                six=np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
                for i in range(6):
                    six[i]=approx[i][0]
                edgesum=[0,0,0,0,0,0]
                for i in range(6):  
                    left=(i-1+6)%6
                    right=(i+1)%6
                    edgesum[i]=(six[i][0]-six[right][0])**2+(six[i][1]-six[right][1])**2+(six[i][0]-six[left][0])**2+(six[i][1]-six[left][1])**2
                index=edgesum.index(max(edgesum))  #marker point 1
                #print edgesum,sum(edgesum)/cv2.contourArea(cont)
                if max(edgesum)<100000 and sum(edgesum)/cv2.contourArea(cont)<20:
                    for i in range(6):
                        sixset[i]=six[(index+i)%6]
                    count6=1  #we have seen the six edges marker
                    img=cv2.polylines(img,[sixset],True,(0,0,255),3)
                    cv2.circle(img,(sixset[0][0],sixset[0][1]),10,(55,255,155),8)
                    imagepoints[0][0][0]=sixset[0]         #first marker point
                    #print "sixset",sixset
           ################################### Locate tri  ###################################         
            if (len(approx)==3):       
                tri=np.array([[0,0],[0,0],[0,0]])
                for i in range(3):
                    tri[i]=approx[i][0]
                    # if len(tri_info)!=2:
                        # tri_info.append(tri)
                img=cv2.polylines(img,[tri],True,(255,255,29),3)
            ################################### Locate rect ###################################    
            if (len(approx)==4):
                rect=np.array([[0,0],[0,0],[0,0],[0,0]])
                for i in range(4):
                    rect[i]=approx[i][0]
                edge1=(rect[0][0]-rect[2][0])**2+(rect[0][1]-rect[2][1])**2#    1   4
                edge2=(rect[1][0]-rect[3][0])**2+(rect[1][1]-rect[3][1])**2#    2   3   
                # if ( edge1<3000 and edge2<3000 ):
                #print (edge1+edge2)/cv2.contourArea(cont)
                if (edge1+edge2)/cv2.contourArea(cont)<20:  
                    # if len(rect_info)!=4:
                        # rect_info.append(rect)
                    img=cv2.polylines(img,[rect],True,(0,0,255),3)
                    if count6==1:       #if we have seen the six edge marker
                        # print "find"
                        vect1=sixset[1]-sixset[0]
                        vect2=sixset[3]-sixset[0]
                        vect3=sixset[5]-sixset[0]
                        awayfrommid=[0,0,0,0]
                        for i in range(4):
                            awayfrommid[i]=(w/2-rect[i][0])**2+(h/2-rect[i][1])**2
                        index=awayfrommid.index(max(awayfrommid))
                        cv2.circle(img,(rect[index][0],rect[index][1]),10,(55,111,255),8)
                        vect=rect[index]-sixset[0]
                        img=cv2.polylines(img,np.array([[sixset[0],sixset[1]]]),True,(22,22,29),3)
                        img=cv2.polylines(img,np.array([[sixset[0],sixset[3]]]),True,(22,22,29),3)
                        img=cv2.polylines(img,np.array([[sixset[0],sixset[5]]]),True,(22,22,29),3)
                        img=cv2.polylines(img,np.array([[sixset[0],rect[index]]]),True,(22,22,29),3)
                        # print (rect[index][0]-w/2)**2+(rect[index][1]-h/2)**2,(h/2)**2+(w/2)**2
                        if (rect[index][0]-w/2)**2+(rect[index][1]-h/2)**2>(h/3.5)**2+(w/3.5)**2:
                            cos1=vect.dot(vect1)/(np.sqrt(vect.dot(vect)*vect1.dot(vect1)))
                            cos2=vect.dot(vect2)/(np.sqrt(vect.dot(vect)*vect2.dot(vect2)))
                            cos3=vect.dot(vect3)/(np.sqrt(vect.dot(vect)*vect3.dot(vect3)))
                            # print "cos1,2,3",cos1, cos2, cos3
                            if cos1>cos2 and cos1>cos3:
                                cv2.circle(img,(rect[index][0],rect[index][1]),20,(55,255,155),8) #marker point 2
                                imagepoints[0][1][0]=rect[index];
                            elif cos2>cos1 and cos2>cos3:
                                cv2.circle(img,(rect[index][0],rect[index][1]),40,(55,255,155),8) #marker point 3
                                imagepoints[0][2][0]=rect[index];
                            elif cos3>cos1 and cos3>cos2:
                                cv2.circle(img,(rect[index][0],rect[index][1]),80,(55,255,155),8) #marker point 4
                                imagepoints[0][3][0]=rect[index];                         
    if count6==1 and imagepoints[0][0][0][0]*imagepoints[0][1][0][0]*imagepoints[0][2][0][0]*imagepoints[0][3][0][0]!=0:
        perspectivepoints[0]=imagepoints[0][0][0]
        perspectivepoints[1]=imagepoints[0][1][0]
        perspectivepoints[2]=imagepoints[0][2][0]
        perspectivepoints[3]=imagepoints[0][3][0]
        PerspectiveMatrix = cv2.getPerspectiveTransform(np.float32(perspectivepoints), np.float32(objectpointsin2d))
        PerspectiveImg = cv2.warpPerspective(imgtopers, PerspectiveMatrix,(480,480))
        print "imgepoint",imagepoints,"objectpoint",objectpoints
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, gray.shape[::-1], None, None) 
        #print ret, mtx, dist, rvecs, tvecs 
        print "RT",rvecs,tvecs
        ##################################### Preprocess the image ####################################\\
        
        pergray=cv2.cvtColor(PerspectiveImg,cv2.COLOR_BGR2GRAY)
        perblurred = cv2.GaussianBlur(pergray, (5, 5), 0)
        perthresh = cv2.adaptiveThreshold(perblurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
        # imshow('s',perthresh)
        percontour = pergray

        ################################### Find contours of rect and tri ####################################
        _,percontours,_ = cv2.findContours(perthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        # PerspectiveImg = cv2.drawContours(PerspectiveImg, percontours, -1, (0,255,0), 1)
       
        imshow('perspective',PerspectiveImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # imshow('imggray',gray)                
    imshow('img',img)
# all_info=rect_info+tri_info
# all_center=[]
# for fig in all_info:
#         center = [sum([fig[i][0] for i in range(len(fig))])/len(fig),sum([fig[i][1] for i in range(len(fig))])/len(fig)]
#         all_center.append(center)
while 1:
    locked=0        
    while True:
        print "RT",rvecs,tvecs
        imgtopers=video_capture.read()[1]
        PerspectiveImg = cv2.warpPerspective(imgtopers, PerspectiveMatrix,(480,480))
        all_info=[]
        img=video_capture.read()[1]
        gray=cv2.cvtColor(PerspectiveImg,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,4)
        _,percontours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        cv2.putText(img, 'Choosing a fig.', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
        #print the located graphs on screen
        
        for percont in percontours:

            # print(cv2.contourArea(cont))
            arc_len = cv2.arcLength(percont,True) 
            approx = cv2.approxPolyDP( percont, 0.03 * arc_len, True )
            if cv2.contourArea(percont)>=200 and cv2.contourArea(percont)<=h*w/2:
                #print len(perapprox)
                x0,y0=perspect(rvecs[0],tvecs[0],mtx,0,0,0)
                if (len(approx)==3):
                    count3=0
                    pertri=np.array([[0,0],[0,0],[0,0]])
                    for i in range(3):
                        pertri[i]=approx[i][0]
                        if (pertri[i][0]>130 and pertri[i][0]<340):
                            count3+=1
                    all_info.append(pertri)
                    edge1=(pertri[0][0]-pertri[1][0])**2+(pertri[0][1]-pertri[1][1])**2
                    edge2=(pertri[0][0]-pertri[2][0])**2+(pertri[0][1]-pertri[2][1])**2
                    edge3=(pertri[1][0]-pertri[2][0])**2+(pertri[1][1]-pertri[2][1])**2
                    # print(edge1/edge2,edge1/edge3,edge2/edge3)
                    # if (edge1/edge2<=1.2 and edge1/edge2>0.8 and edge1/edge3 <= 1.2 and edge1/edge3 > 0.8 and edge2/edge3 <= 1.2 and edge2/edge3 >0.8):               
                    PerspectiveImg=cv2.polylines(PerspectiveImg,[pertri],True,(0,0,255),3)
                if (len(approx)==4):
                    count=0
                    perrect=np.array([[0,0],[0,0],[0,0],[0,0]])
                    for i in range(4):
                        perrect[i]=approx[i][0]
                        if (perrect[i][0]>130 and perrect[i][0]<340):
                            count+=1
                    all_info.append(perrect)
                    edge1=(perrect[0][0]-perrect[2][0])**2+(perrect[0][1]-perrect[2][1])**2#    1   4
                    edge2=(perrect[1][0]-perrect[3][0])**2+(perrect[1][1]-perrect[3][1])**2#    2   3   
                    if ( edge1/edge2<1.2 and edge1/edge2>0.8 and count==4):
                        PerspectiveImg=cv2.polylines(PerspectiveImg,[perrect],True,(0,0,255),3)
        all_center=[]
        for fig in all_info:
            center = [int(sum([fig[i][0] for i in range(len(fig))])/len(fig)),int(sum([fig[i][1] for i in range(len(fig))])/len(fig))]
            all_center.append(center)
        for percont in percontours:
            if cv2.contourArea(percont)>=200 and cv2.contourArea(percont)<=h*w/2:
                # print(cv2.contourArea(cont))
                arc_len = cv2.arcLength(percont,True) 
                approx = cv2.approxPolyDP( percont, 0.03 * arc_len, True )
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
                    if True:#(max(edge)/min(edge)<2.3 and max(edge)/min(edge)>1.7):
                        # print penta
                        cv2.polylines(PerspectiveImg,[penta],True,(233,100,100),3)
                        distance=[]
                        for i in range(len(all_center)):
                            center=all_center[i]
                            # distance.append(np.sqrt((center[0]-penta[0][0])**2+(center[1]-penta[0][1])**2))
                            if (np.sqrt((center[0]-penta[0][0])**2+(center[1]-penta[0][1])**2)<50):
                                cv2.putText(img, 'This one?', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155,175,131), thickness = 1, lineType = 8) 
                                cv2.polylines(PerspectiveImg,[all_info[i]],True,(255,255,122),3)
                                locked_fig=all_info[i]
                                original_center = np.array(center)
                                locked=1
                                break            
        imshow('perspective',PerspectiveImg) 
        imshow('img',img)
        # print "locked",locked
        if cv2.waitKey(1) & 0xFF == ord('q') and locked==1:
            break

    while True:
        img=video_capture.read()[1]
        imgtopers=video_capture.read()[1]
        # img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,4)
        # imshow('s',thresh)
        contour = gray
        image,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        # for fig in all_info:
        #     cv2.polylines(img,[fig],True,(123,255,100),3)
        for cont in contours:
            arc_len = cv2.arcLength(cont,True) 
            approx = cv2.approxPolyDP( cont, 0.03 * arc_len, True )
            if cv2.contourArea(cont)>=200 and cv2.contourArea(cont)<=h*w/2:
                if (len(approx)==3):       
                        tri=np.array([[0,0],[0,0],[0,0]])
                        for i in range(3):
                            tri[i]=approx[i][0]
                            # if len(tri_info)!=2:
                                # tri_info.append(tri)
                        img=cv2.polylines(img,[tri],True,(255,255,29),3)
                    ################################### Locate rect ###################################    
                if len(approx)==4:
                    rect=np.array([[0,0],[0,0],[0,0],[0,0]])
                    for i in range(4):
                        rect[i]=approx[i][0]
                    edge1=(rect[0][0]-rect[2][0])**2+(rect[0][1]-rect[2][1])**2#    1   4
                    edge2=(rect[1][0]-rect[3][0])**2+(rect[1][1]-rect[3][1])**2#    2   3   
                    # if ( edge1<3000 and edge2<3000 ):
                    #print (edge1+edge2)/cv2.contourArea(cont)
                    if (edge1+edge2)/cv2.contourArea(cont)<20:  
                        # if len(rect_info)!=4:
                            # rect_info.append(rect)
                        img=cv2.polylines(img,[rect],True,(0,0,255),3)
                ########################## find the nearest fig #####################
                if len(approx)==5:
                    penta=np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])
                    for i in range(5):
                        penta[i]=approx[i][0]
                    edge=[]
                    edge.append((penta[0][0]-penta[1][0])**2+(penta[0][1]-penta[1][1])**2)
                    edge.append((penta[1][0]-penta[2][0])**2+(penta[1][1]-penta[2][1])**2)
                    edge.append((penta[2][0]-penta[3][0])**2+(penta[2][1]-penta[3][1])**2)
                    
                    if (max(edge)/min(edge)<2.5 and max(edge)/min(edge)>0.8):
                        img=cv2.polylines(img,[penta],True,(244,100,100),3)
                        # distance.append(np.sqrt((center[0]-penta[0][0])**2+(center[1]-penta[0][1])**2))
                        # print "penta[0][1]",penta[0][1],"center[1]",center[1]
                        height=-penta[0][1]+center[1]+100
                        # height=-f_h(rvecs[0],tvecs[0],mtx,penta[0],center)
                        ##################### draw the tri in 3D############################
                        print "tri or rect?",len(locked_fig)
                        if len(locked_fig)==3:
                            print "tirheight",height   
                            x1,y1=perspect(rvecs[0],tvecs[0],mtx,locked_fig[0][0],locked_fig[0][1],0)
                            x2,y2=perspect(rvecs[0],tvecs[0],mtx,locked_fig[1][0],locked_fig[1][1],0)
                            x3,y3=perspect(rvecs[0],tvecs[0],mtx,locked_fig[2][0],locked_fig[2][1],0)  
                            x1h,y1h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[0][0],locked_fig[0][1],-height)
                            x2h,y2h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[1][0],locked_fig[1][1],-height)
                            x3h,y3h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[2][0],locked_fig[2][1],-height)
                            if x1h!=np.nan and x2h!=np.nan and x3h!=np.nan: 
                                after_projection_trih=np.array([[int(x1h),int(y1h)],[int(x2h),int(y2h)],[int(x3h),int(y3h)]])
                                cv2.polylines(img,[after_projection_trih],True,(255,0,155),3)
                                cv2.polylines(img,[np.array([[int(x1h),int(y1h)],[x1,y1]])],True,(255,0,155),3)
                                cv2.polylines(img,[np.array([[int(x2h),int(y2h)],[x2,y2]])],True,(255,0,155),3)
                                cv2.polylines(img,[np.array([[int(x3h),int(y3h)],[x3,y3]])],True,(255,0,155),3)
                            
                        ##################### draw the rect in 3D############################
                        if len(locked_fig)==4:  
                            print "rectheight",height  
                            x1,y1=perspect(rvecs[0],tvecs[0],mtx,locked_fig[0][0],locked_fig[0][1],0)
                            x2,y2=perspect(rvecs[0],tvecs[0],mtx,locked_fig[1][0],locked_fig[1][1],0)
                            x3,y3=perspect(rvecs[0],tvecs[0],mtx,locked_fig[2][0],locked_fig[2][1],0)
                            x4,y4=perspect(rvecs[0],tvecs[0],mtx,locked_fig[3][0],locked_fig[3][1],0)
                            x1h,y1h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[0][0],locked_fig[0][1],-height)
                            x2h,y2h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[1][0],locked_fig[1][1],-height)
                            x3h,y3h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[2][0],locked_fig[2][1],-height)
                            x4h,y4h=perspect(rvecs[0],tvecs[0],mtx,locked_fig[3][0],locked_fig[3][1],-height)
                            if x1h!=np.nan and x2h!=np.nan and x3h!=np.nan and x4h!=np.nan: 
                                after_projection_recth=np.array([[int(x1h),int(y1h)],[int(x2h),int(y2h)],[int(x3h),int(y3h)],[int(x4h),int(y4h)]])
                                cv2.polylines(img,[after_projection_recth],True,(255,155,0),3)
                                cv2.polylines(img,[np.array([[int(x1h),int(y1h)],[x1,y1]])],True,(255,155,0),3)
                                cv2.polylines(img,[np.array([[int(x2h),int(y2h)],[x2,y2]])],True,(255,155,0),3)
                                cv2.polylines(img,[np.array([[int(x3h),int(y3h)],[x3,y3]])],True,(255,155,0),3)
                                cv2.polylines(img,[np.array([[int(x4h),int(y4h)],[x4,y4]])],True,(255,155,0),3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # imshow('imggray',gray)                
        imshow('img',img)

cv2.destroyAllWindows()