import os
import shutil

img_dir = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train/arg"
label_dir = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/labels/train"

output_dir = img_dir  # mesmo destino

# pega todos arquivos de imagem na pasta arg
img_files = os.listdir(img_dir)

for img_file in img_files:
    # remove extensão (.png, .jpg, etc.)
    name_without_ext = os.path.splitext(img_file)[0]

    label_file = name_without_ext + ".txt"
    label_path = os.path.join(label_dir, label_file)

    # se label existir, copia
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_dir, label_file))

        print(f"Copiado: {label_file}")