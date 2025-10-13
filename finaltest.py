# test6
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import csv
import os
import time 
import torch
import  tensorflow as tf
import psutil
print(tf.config.list_physical_devices('GPU'))


save_path = "/home/user1/MohamedAbdeltawabsThesis"
os.makedirs(save_path, exist_ok=True)
# flag2 for wheather we ar in first place or second , flag unused till now, models and getvv camera ready
flag2=False
flag=False
model = YOLO("/home/user1/MohamedAbdeltawabsThesis/JSON2YOLO/runs/segment/train7/weights/best.pt") 
model2= YOLO("yolov8n.pt")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
process = psutil.Process(os.getpid())
frame_count = 0

#first we segment and search to the point (not flag2) - if point found flag2=true then segment and
#  update the point for any exceptions , error , wrong values return to flag2false to find new point , either wise finish with object detection#
while(1):
    start = time.time()
    frame_count += 1


    if not flag2:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        frame = np.asanyarray(color_frame.get_data())
        depth= np.asanyarray(depth_frame.get_data())
        prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detect=model.predict(source=frame ,conf=0.5)
        detected0=detect[0]
        detectf=detected0.plot()
        list1=detected0.boxes.cls
        if len(list1)==0 :
            print("nothing is here yet")
        else:
            for i,object in enumerate(detected0.boxes) :
                if object.cls ==0.0:
                    xyupavment=object.xyxy[0].tolist()
                    ypavment= xyupavment[3]-(((xyupavment[3]-xyupavment[1]))/2)
                    maxx=0
                    minx=6000
                    xs=[] 
                    xh=[]
                    xl=[]
                    # for point in (detected0.masks[i].xy[0]):
                    #         x=int(point[0])
                    #         xs.append(x)
                    #         if x>maxx:
                    #                 maxx=x
                    #         if x<minx:
                    #                 minx=x
                        
                    #         medx=(minx+maxx)/2
                    #         for x in xs:
                    #                 if x>medx:
                    #                         xh.append(x)
                    #                 if x<medx:
                    #                         xl.append(x)
                    #         averageh = ( np.nanmean(xh))
                    #         averagel =( np.nanmean(xl)) 
                    #         xpavment= (averagel+(averageh-averagel)/2)
                    # xpavment=(xyupavment[3]+xyupavment[0])/1.75
                    # x1=int(xpavment)
                    # y1=int(ypavment)
                    # x2=int(xyupavment[2])
                    # y2=int(xyupavment[3])
                    

                    # cv.rectangle(detectf, (x1, y1), (x2,y2), (255, 255, 255), -1)
                    # maxx=0
                    # minx=6000
                    # xs=[] 
                    # xh=[]
                    # xl=[]
                    # for point in (detected0.masks[i].xy[0]):
                    #         x=int(point[0])
                    #         xs.append(x)
                    #         if x>maxx:
                    #                 maxx=x
                    #         if x<minx:
                    #                 minx=x
                        
                    #         medx=(minx+maxx)/2
                    #         for x in xs:
                    #                 if x>medx:
                    #                         xh.append(x)
                    #                 if x<medx:
                    #                         xl.append(x)
                    #         averageh = ( np.nanmean(xh))
                    #         averagel =( np.nanmean(xl)) 
                    #         xpavment= (averagel+(averageh-averagel)/2)
                    xs = [int(point[0]) for point in detected0.masks[i].xy[0]]
                    minx = min(xs)
                    maxx = max(xs)
                    medx = (minx + maxx) / 2

                    xs = np.array(xs)
                    xl = xs[xs < medx]
                    xh = xs[xs > medx]

                    averagel = np.nanmean(xl)
                    averageh = np.nanmean(xh)
                    xpavment = averagel + ((averageh - averagel) / 2)
                    print(xpavment,ypavment)
                    fixed_point=[[xpavment,ypavment]]
                    prev_pts = np.array([fixed_point], dtype=np.float32)
                    flag2=True
                    # 
                    xx=int(xpavment)
                    yy=int(ypavment)
                    cv.circle(detectf, (xx, yy), radius=5, color=(255, 0, 0), thickness=-1)
            
                if object.cls ==3.0:
                    print("university pavements are here")
                    xyupavment=object.xyxy[0].tolist()
                    ypavment= xyupavment[3]-(((xyupavment[3]-xyupavment[1]))/2)
                    maxx=0
                    minx=6000
                    xs=[] 
                    xh=[]
                    xl=[]
                    # for point in (detected0.masks[i].xy[0]):
                    #         x=int(point[0])
                    #         xs.append(x)
                    #         if x>maxx:
                    #                 maxx=x
                    #         if x<minx:
                    #                 minx=x
                        
                    #         medx=(minx+maxx)/2
                    #         for x in xs:
                    #                 if x>medx:
                    #                         xh.append(x)
                    #                 if x<medx:
                    #                         xl.append(x)
                    #         averageh = ( np.nanmean(xh))
                    #         averagel =( np.nanmean(xl)) 
                    #         xpavment= (averagel+(averageh-averagel)/2)
                    # xpavment=(xyupavment[3]+xyupavment[0])/1.75
                    # x1=int(xpavment)
                    # y1=int(ypavment)
                    # x2=int(xyupavment[2])
                    # y2=int(xyupavment[3])
                    

                    # cv.rectangle(detectf, (x1, y1), (x2,y2), (255, 255, 255), -1)
                    # maxx=0
                    # minx=6000
                    # xs=[] 
                    # xh=[]
                    # xl=[]
                    # for point in (detected0.masks[i].xy[0]):
                    #         x=int(point[0])
                    #         xs.append(x)
                    #         if x>maxx:
                    #                 maxx=x
                    #         if x<minx:
                    #                 minx=x
                        
                    #         medx=(minx+maxx)/2
                    #         for x in xs:
                    #                 if x>medx:
                    #                         xh.append(x)
                    #                 if x<medx:
                    #                         xl.append(x)
                    #         averageh = ( np.nanmean(xh))
                    #         averagel =( np.nanmean(xl)) 
                    #         xpavment= (averagel+(averageh-averagel)/2)
                    xs = [int(point[0]) for point in detected0.masks[i].xy[0]]
                    minx = min(xs)
                    maxx = max(xs)
                    medx = (minx + maxx) / 2

                    xs = np.array(xs)
                    xl = xs[xs < medx]
                    xh = xs[xs > medx]

                    averagel = np.nanmean(xl)
                    averageh = np.nanmean(xh)
                    xpavment = averagel + ((averageh - averagel) / 2)
                    print(xpavment,ypavment)
                    fixed_point=[[xpavment,ypavment]]
                    prev_pts = np.array([fixed_point], dtype=np.float32)
                    flag2=True
                    # 
                    xx=int(xpavment)
                    yy=int(ypavment)
                    cv.circle(detectf, (xx, yy), radius=5, color=(255, 0, 0), thickness=-1)
            

    #                 break
            
    else:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = np.asanyarray(color_frame.get_data())
        depth= np.asanyarray(depth_frame.get_data())
        detect=model.predict(source=frame ,conf=0.5)
        detected0=detect[0]
        detectf=detected0.plot()
        # cv.rectangle(detectf, (x1, y1), (x2,y2), (255, 255, 255), -1)

        next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        x, y = map(int,next_pts[0][0])
        x_int, y_int = int(x), int(y)
        if y>430 :
                flag2=False
                continue
        # ---------- TEST----------#
        list2=detected0.boxes.cls

        try :
            if  (0.0 or 3.0) in list2:
                if (0.0) in list2:
                    index= list2.index(0.0)
                if (3.0) in list2:
                    index= list2.index(3.0)             
                list3=detected0.boxes[index].xyxy[0]

                if (x<list3[0]+50):
                    flag2=False
                    continue
                if (x>list3[2]):
                    flag2=False
                    continue
        except Exception:
            pass

        # ---------- TEST finished ----------#

        
        try:
            depthpoint = depth_frame.get_distance(x, y)
        except Exception:
            depthpoint=0
            flag2=False
        else:
            cv.circle(detectf, (x_int, y_int), radius=5, color=(0, 255, 0), thickness=-1)
            prev_pts = next_pts
            prev_gray = gray
        finally:
            if depthpoint==0 :
                flag2=False


    # car and person detection
    detect2=model2.predict(source=detectf,conf=0.75,classes=[0,2])
    detectz=detect2[0].plot()
    list1=(detect2[0].boxes.xyxy).tolist()
        
    for list in list1 :
        hi=(list[0]+list[2])/2
        bi=(list[1]+list[3])/2
        hibi=(int(hi),int(bi))
        distance=int(depth_frame.get_distance(int(hi),int(bi)))
        dis=str(distance)
        cv.putText(detectz,dis,hibi,cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
    end = time.time()
    exec_time = end - start
    print(frame_count)
    print(f"Execution Time: {exec_time:.4f} seconds")
    cpu_usage = process.cpu_percent()
    mem_usage = process.memory_info().rss / 1024 ** 2  # in MB
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {mem_usage:.2f} MB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print("---------------------------------------------------------------------------------------------------------")
    filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")    
    cv.imshow("segment",detectz)
    cv.imwrite(filename,detectz)

        # os.makedirs(detectz, exist_ok=True)
        # out.write(detectf)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
pipeline.stop()
cv.destroyAllWindows

    






