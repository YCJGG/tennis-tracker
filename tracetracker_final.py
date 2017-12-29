import cv2
import numpy as np
import math
from argparse import ArgumentParser

def cos_func(a, b, c):
    vector_1 = [b[0] - a[0], b[1] - a[1]]
    vector_2 = [c[0] - b[0], c[1] - b[1]]
    length_1 = math.sqrt(vector_1[0]**2 + vector_1[1]**2)
    length_2 = math.sqrt(vector_2[0]**2 + vector_2[1]**2)
    return float(vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]) / length_1 / length_2

def put_time(flag_f):
    time = flag_f*1000//60
    time_ms = time % 1000
    time_s  =  time // 1000
    time_m = time_s // 60
    time_ms_str = time_ms
    if time_s < 10:
        time_s = '0'+str(time_s)
    if time_m < 10:
        time_m ='0' + str(time_m)
    if time_ms < 10:
        time_ms_str = '00'+str(time_ms)
    if time_ms < 100 and time_ms >= 10 :
        time_ms_str = '0' + str(time_ms)
    return str(time_m)+':'+str(time_s)+':'+str(time_ms_str)

def get_velocity(r,pre_r,index, pre_index,point,prepoint,theta):
    r_baseball = min(r,pre_r)
    time = (index - pre_index)*(1/60)
    r_baseball_real = 0.065
    depth_estimate_costant = 1 / math.cos(theta/180*math.pi)
    dis = math.sqrt((point[1]-prepoint[1])**2+(point[0]-prepoint[0])**2)*depth_estimate_costant
    #print(dis, r_baseball, time)
    v = dis*r_baseball_real/r_baseball /time * 3.6
    v = round(v,2)
    if v >= 200:
        v = 'NAN'
    return str(v)

def write2file(text, text0,text3):
    file = open('content.txt','a')
    file.write('time: '+text0+' '+'coordinate: '+text3+' '+text+'\n')
    file.close()

def putTextWithShadow(frame,text,position,size,color,thickness):
    font = cv2.FONT_HERSHEY_DUPLEX 
    cv2.putText(frame,text,(position[0]+1,position[1]+2), font, size,(0,0,0),thickness,cv2.LINE_AA)
    cv2.putText(frame,text,position, font, size,color,thickness,cv2.LINE_AA)

def putAllText(frame,text,position,color):
    putTextWithShadow(frame,text,position,0.8,color,1)

def detect_video(video,center,coor1,area_f1,area_f2,rate_f1,rate_f2,x_d,y_d):
    camera = cv2.VideoCapture(video)
    history = 20   
    # bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # knn
    bs = cv2.createBackgroundSubtractorMOG2(history=history)  
    bs.setHistory(history)
    #bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=history)

    # Define the codec and create VideoWriter object
    #fps = camera.get(cv2.CAP_PROP_POS_FRAMES)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #print(fps,width,height)
    out = cv2.VideoWriter('output.avi',-1,60,(int(width),int(height)))

    frames = 0 
    temp = None
    flag_f = 0
    flag = 0
    prev_points = [] #Points at t-1
    curr_points = [] #Points at t
    lines=[] #To keep all the lines overtime
    velocity_ave=[]
    flag_index = 0
    flag_k =1
    flag_kk =0
    while True:
        # calculate the time
        t_s = cv2.getTickCount()
        
        ct = []
        res, frame = camera.read()
        frame_copy = frame
        if not res:
            break
        t1 = cv2.getTickCount()
        fg_mask = bs.apply(frame)   #  foreground mask
        t2 = cv2.getTickCount()
        if frames < history:
            frames += 1
            continue
        
        # 
        th = cv2.threshold(fg_mask.copy(), 250, 255, cv2.THRESH_BINARY)[1]
        
        erode = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        
        dilated = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
          
        flag_f += 1
  
        img = dilated               
             
        # get the rectangles
        image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # to decrease the complexity
        if flag== 0 :
            rate_f = rate_f1
            area_f = area_f1
            flag = 1
        if flag == 1 :
            rate_f = rate_f2
            area_f = area_f2


        for c in contours:
            # coordinate
            x, y, w, h = cv2.boundingRect(c)
            # area
            area = cv2.contourArea(c)
            noneZero = 0
            for i in img[y:y + h]:
                noneZero += cv2.countNonZero(i[x:x + w])
            rate = float(noneZero) / w / h
            if   rate > rate_f and rate < 1 and w*h>area_f:

                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # get the coordiate of center
                k = [int(x+w/2),int(y+h/2),(min(h,w)),flag_f]

                ct.append(k)
        center.append(ct)
        #cv2.circle(frame,(cx,cy),10,255,-1)
       
        if flag_f is 1:
            coor1 = [ [x] for x in center[0] ]

        #for fra in center[1:] :
        #coor1 = [ [x] for x in frame ]
        fra = ct
        if flag_f is 1:
            continue
        for co in coor1 :
            #cx=[]
            #cy=[]  
            xy=[]    
            for iterm in fra:
                x = int(co[-1][0])
                y = int(co[-1][1])
                if len(co) == 1:
                    [x1, y1] = [x - 1, y]
                else:
                    [x1, y1] = co[-2][0:2]
                #area = int(co[-1][2])
                xd = abs(int(co[-1][0])-int(iterm[0]))
                yd = abs(int(co[-1][1])-int(iterm[1]))
                if  ((abs(int(co[-1][0])-int(iterm[0])) > 3) or(abs(int(co[-1][1])-int(iterm[1]))>5)) and xd<=x_d and yd<=y_d:
                    #1.rmvb xd=50 yd=40 area_f=25 
                    #2.rmvb xd=60(58) yd=45 area_f=60 （48.5）
                    
                    xy.append(iterm[0:4])
                    #cv2.circle(frame,(int(iterm[0]),int(iterm[1])),4,150,-1)
            if len(xy) == 0:
                continue

            #method2 costheta  direction vector
            cos_array = [abs(cos_func([x1, y1], [x, y], point[0:2])) for point in xy]
            cos_max = max(cos_array)
            max_point_index = cos_array.index(cos_max)
            co.append(xy[max_point_index])
            
            #cv2.circle(frame,(co[-1][0], co[-1][1]),10,[255, 255, 255],-1)
        m = 0
        xx2 = []
        for it in coor1[flag_index]:
            xx2.append(int(it[0]))
        std_max = np.std(xx2)

        for line in coor1:
            xx1 = []
            for it in line:
                xx1.append(int(it[0]))
        if np.std(xx1) > std_max:
            flag_k=0
        if flag_k==1:
            flag_kk = flag_kk+1
        if flag_kk >=20:
            coor1=[coor1[flag_index]]

        for line_index in range( len(coor1)):
            xx=[]
            for it in coor1[line_index]:
                xx.append(int(it[0]))
            if np.std((xx))>=m :
                m=np.std((xx))
                mm = coor1[line_index]
                flag_index = line_index
        
        if flag_f >= 20:
            co_x = int(mm[-1][0])
            co_y = int(mm[-1][1])
            co_r =  mm[-1][2]
            co_index =  int(mm[-1][3])
            point = (co_x,co_y)
            prepoint = point
            if len(mm)>=2:
                prepoint = (int(mm[-2][0]),int(mm[-2][1]))
                pre_co_r =  mm[-2][2]
                pre_co_index =  int(mm[-2][3])
            #cv2.circle(frame, prepoint, 10, (255,255,255))
            cv2.circle(frame, point, 10, (0,0,255),-1)
            cv2.line(frame, prepoint, point, (0,0,0),3)
            lines.append((prepoint,point))
            for (pt1, pt2) in lines:
                cv2.line(frame, pt1, pt2, (255,0,0),3)
            velocity = get_velocity(co_r,pre_co_r,co_index,pre_co_index,point,prepoint,theta=83.5)
            if velocity != 'NAN':       
                velocity_ave.append(float(velocity))
            v_average = np.mean(velocity_ave)
            v_average = round(v_average)
            v_max = max(velocity_ave)
            #print(co_r)  
            #put text
            text4 = put_time(flag_f)
            text0 = 'velocity:'+' '+str(velocity)+' km/h'
            
            text1 = 'velocity_mean: '+str(v_average)+' km/h'
            text2 = 'velocity_max: '+str(v_max)+' km/h'
            #text1 = 'X: '+str(co_x)
            #text2 = 'Y: '+str(co_y)
            text3 = '('+str(co_x)+','+str(co_y)+')' 
            text=[text2,text1,text0,text4,text3]
            #cv2.rectangle(frame, (1380, 730), (1700, 900), (0, 0, 0), -1)
            position = (1400,769)
            color = (0,255,255)
            for i in range(len(text)):
                if i > 2:
                    color = (255,100,0)
                putTextWithShadow(frame,text[i],(position[0],position[1]+i*30),0.8,color,1)
            write2file(text4,text0,text3)
            # frame write to files
            out.write(frame)
        cv2.imshow("detection", frame)
        #t_end = cv2.getTickCount()
        #t = (t_end - t_s)/cv2.getTickFrequency()
        #t_main = (t2-t1)/cv2.getTickFrequency()
        #t_main_ratio = t_main / t
        #print (t,t_main,t_main_ratio*100)
        #print ('\n')
        #cv2.imshow('curve',frame_copy)
        #cv2.imshow("back", img)
        #cv2.imshow("no-diff", dilated)
        if cv2.waitKey(110) & 0xff == 27:
           break
    camera.release()
    out.release()

def build_parser():
    parser = ArgumentParser()
    requiredArgs = ['video']
    args = ['rate_f1', 'rate_f2', 'area_f1', 'area_f2', 'x_d', 'y_d']
    for arg in args:
        parser.add_argument('--' + arg,
            dest=arg,
            metavar=arg.upper(), required=False)
    for arg in requiredArgs:
        parser.add_argument('--' + arg,
            dest=arg,
            metavar=arg.upper(), required=True)
    return parser

if __name__ == '__main__':
    center=[]
    coor1=[]
    parser = build_parser()
    options = parser.parse_args()
    video = options.video or '1.mp4'
    rate_f1 = float(options.rate_f1 or 0.4)
    area_f1 = float(options.area_f1 or 800)
    rate_f2 = float(options.rate_f2 or 0.2)
    area_f2 = float(options.area_f2 or 25)
    x_d = float(options.x_d or 50)
    y_d = float(options.y_d or 40)
    detect_video(video,center,coor1,area_f1,area_f2,rate_f1,rate_f2,x_d,y_d)