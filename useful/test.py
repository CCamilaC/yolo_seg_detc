import cv2
import numpy as np
import math
import os

# Caminhos
input_folder = "/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/train/"
output_folder = os.path.join(input_folder, "results")

os.makedirs(output_folder, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg", ".bmp")
files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]

print(f"{len(files)} imagens encontradas na pasta train.")

# ------------------------------
# FUNÇÃO: ANGULO DE UMA LINHA
# ------------------------------
def line_angle(line):
    x1, y1, x2, y2 = line[0]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

# ------------------------------
# PROCESSAR IMAGENS
# ------------------------------
for filename in files:

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: {filename}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ----------------------
    # 1. COR (CANDIDATOS)
    # ----------------------
    lower_green = np.array([20, 60, 40])
    upper_green = np.array([100, 255, 255])


    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    name, ext = os.path.splitext(filename)


    mask_save_path = os.path.join(output_folder, name + "_mask.png")
    cv2.imwrite(mask_save_path, mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        continue

    min_area = 2000
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
    
    # ------------------------------
    # PROCESSO POR CONTORNO
    # ------------------------------
    for cnt in filtered_contours:

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # ----------------------
        # 2. FORMA GLOBAL
        # ----------------------
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        # ----------------------
        # 3. RECORTE
        # ----------------------
        roi = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ----------------------
        # CANNY + HOUGH
        # ----------------------
        edges = cv2.Canny(gray, 100, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )


        # ----------------------
        # DECISÃO FINAL MELHORADA
        # ----------------------
        is_probe = (aspect_ratio > 3 )

        if not is_probe:
            continue

        # ----------------------
        # DESENHAR RESULTADO
        # ----------------------
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "probe", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ----------------------
        # CENTRO
        # ----------------------
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        # ----------------------
        # PONTA
        # ----------------------
        pts = cnt.reshape(-1, 2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        cone_tip = tuple(pts[np.argmax(dists)])

        cv2.circle(img, cone_tip, 8, (255, 0, 0), -1)
        cv2.putText(img, "Cone", (cone_tip[0] + 10, cone_tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # ----------------------
        # ORIENTAÇÃO
        # ----------------------
        dx = cone_tip[0] - cx
        dy = cone_tip[1] - cy

        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        cv2.putText(img, f"{angle:.1f} deg", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ------------------------------
    # SALVAR
    # ------------------------------
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, img)

    print(f"Processado e salvo: {save_path}")

print("\nFinalizado! Todas as imagens foram processadas.")