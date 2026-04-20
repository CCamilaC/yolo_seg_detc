import os

folder_path = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train/arg"

output_file = os.path.join(folder_path, "arquivos_listados.txt")

files = os.listdir(folder_path)

with open(output_file, "w") as f:
    for file in files:
        f.write(file + "\n")

print(f"Lista salva em: {output_file}")