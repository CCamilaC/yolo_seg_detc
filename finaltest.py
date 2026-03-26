import cv2 as cv
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
import time
import psutil
import torch

# Save path for frames
save_path = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/JSON2YOLO/runs/segment/train2/"
os.makedirs(save_path, exist_ok=True)

flag2 = False

# Load YOLOv8 **detection** weights
model = YOLO(
    "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/JSON2YOLO/runs/segment/train2/weights/best(1).pt"
)
model.to("cpu")

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
pipeline.start(config)

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
process = psutil.Process(os.getpid())
frame_count = 0

prev_pts = None
prev_gray = None


while True:
    start = time.time()
    frame_count += 1

    # Get frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    frame = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ---- YOLO DETECTION (conf >= 0.8) ----
    results = model.predict(frame, conf=0.7)
    result = results[0]
    detect_img = result.plot()

    boxes = result.boxes

    if not flag2:
        if len(boxes) == 0:
            print("Nothing detected yet")
        else:
            for obj in boxes:

                # Skip low-confidence detections
                if float(obj.conf) < 0.8:
                    continue

                cls_id = int(obj.cls)
                if cls_id in [0, 3]:  # pavement classes
                    x1, y1, x2, y2 = obj.xyxy[0].tolist()

                    # Midpoint of bounding box
                    xpavement = (x1 + x2) / 2
                    ypavement = (y1 + y2) / 2

                    print(xpavement, ypavement, "Conf:", float(obj.conf))

                    prev_pts = np.array([[[xpavement, ypavement]]], dtype=np.float32)
                    prev_gray = gray.copy()
                    flag2 = True

                    cv.circle(detect_img, (int(xpavement), int(ypavement)),
                              radius=5, color=(255, 0, 0), thickness=-1)
    else:
        # ---- Optical Flow Tracking ----
        next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        x, y = map(int, next_pts[0][0])

        prev_pts = next_pts
        prev_gray = gray

        if y > 430:
            flag2 = False
            continue

        try:
            depthpoint = depth_frame.get_distance(x, y)
        except Exception:
            depthpoint = 0
            flag2 = False
        else:
            cv.circle(detect_img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        finally:
            if depthpoint == 0:
                flag2 = False

    # Display and save
    filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
    cv.imshow("detect", detect_img)
    cv.imwrite(filename, detect_img)

    end = time.time()
    print(frame_count, f"Execution Time: {end - start:.4f} sec")
    print(f"CPU Usage: {process.cpu_percent():.2f}%")
    print(f"Memory Usage: {process.memory_info().rss / 1024**2:.2f} MB")

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv.destroyAllWindows()
