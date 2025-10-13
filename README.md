# RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Genration-Dynamic-Object-Detection


# Sensor Used
Realsense D435i Camera

# Pre-requisites
- **ROS 2 Humble** ([installation guide](https://docs.ros.org/en/humble/index.html))
- **Intel RealSense ROS2 Wrapper** ([repository](https://github.com/IntelRealSense/realsense-ros))
- **YOLOv8** (install via Ultralytics [yolov8 installation docs](https://docs.ultralytics.com/))

# Training the Segmentation Model

1. Create a segmentation project on [Roboflow Universe](https://universe.roboflow.com/).
2. Upload, annotate, and augment dataset images.
3. Download the annotated data as a zip file.
4. Organize the dataset as follows:
    - 80% images: `/coco/images/train`
    - 10% images: `/coco/images/valid`
    - 10% images: `/coco/images/test`
5. Update file paths in `data.yaml` accordingly.


# Use the following Command to start the Training
```bash
yolo task=detect mode=train model=yolov8n.pt data=/path_to/coco/data.yaml

```

# Implementation of Pavement Segmentation based Center Point Generation

In finaltest.py update the path location (model2) to the Segmentation weight file 


Connect the RealSense Camera and run the python file

```bash
python3 finaltest.py

```
Dynamic object detection uses the YOLOv8n model to detect persons and cars in real time.

# Final Result

![Segment4](https://github.com/user-attachments/assets/3627cb2a-cf88-47e1-9b64-ceeef9b67baa)


