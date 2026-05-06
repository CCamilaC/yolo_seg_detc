# Probe Detection Task

This repository is based on this one: [RealSense Camera Based Pavement Segmentation and Centre Point Generation Dynamic Object Detection](https://github.com/swarajtendulkar10/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection.git)

Autor: Swaraj Tendulkar

## Informations 

If you want to train the model on Google Colab using an external GPU, you can use the `yolo_seg_detc.ipynb` notebook, adjusting the necessary parameters.

If you need more images, you can find them in the [Yolo_Probe_Detection](https://github.com/CCamilaC/Yolo_Probe_Detector.git) repository.

A manual augmentation was added to change the color of the detected object (in this case, the Probe). For this, I used the `argumentation.py` script located in the `useful` folder.

The `useful` folder also contains additional scripts that help with image verification and label adjustment.

## Classical Vision Codes

The `classical_vision.py` script is already implemented to connect to the RealSense camera, while `classical_vision_photos.py` is used to test using the photos in the dataset.

### Importante 
It is important to use a USB 3.0 port (you can verify it by checking the inside of the connector, which is blue). If you use a USB 2.0 port, you may encounter an error like the following in the code:
```bash
RuntimeError: Frame didn't arrive within 5000
```
## Difference Between USB 2.0 and USB 3.0

The main difference between USB 2.0 and USB 3.0 is the speed and data transfer capability.

- **USB 2.0:** up to 480 Mbps (slower)
- **USB 3.0:** up to 5 Gbps (more than 10x faster)


