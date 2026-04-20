import os
import random
import string

folder = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train/arg"

files = os.listdir(folder)

# pega nomes base (sem extensão)
base_names = [os.path.splitext(f)[0] for f in files]

# cria mapa: nome base -> letra aleatória fixa
mapping = {}

letters = list(string.ascii_lowercase)

for name in set(base_names):
    mapping[name] = random.choice(letters)

# renomeia arquivos
for file in files:
    old_path = os.path.join(folder, file)

    name, ext = os.path.splitext(file)

    if name in mapping:
        new_name = name + "_" + mapping[name] + ext
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

        print(f"{file} -> {new_name}")