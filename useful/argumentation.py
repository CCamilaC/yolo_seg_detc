import os
import cv2
import torch
import random
import kornia.color as KC

input_dir = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train"
output_dir = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train/arg"

os.makedirs(output_dir, exist_ok=True)

def yellow_to_green_hsv(img):
    # OpenCV -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # BxCxHxW

    hsv = KC.rgb_to_hsv(img)

    shift = random.uniform(-20/360, -60/360)
    hsv[:, 0] = (hsv[:, 0] + shift) % 1.0

    img_out = KC.hsv_to_rgb(hsv)

    img_out = img_out.squeeze(0).permute(1, 2, 0).numpy()
    img_out = (img_out * 255).astype("uint8")

    # RGB -> BGR
    img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    return img_out


for filename in os.listdir(input_dir):
    if not filename.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    aug_img = yellow_to_green_hsv(img)

    new_name = filename.replace(".jpg", "_green.jpg")
    cv2.imwrite(os.path.join(output_dir, new_name), aug_img)