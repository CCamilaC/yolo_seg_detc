import cv2
import numpy as np
import math

# Carregar imagem
img = cv2.imread('/home/camila/yolo_detc_seg/RealSense-Camera-Based-Pavement-Segmentation-and-Centre-Point-Generation-Dynamic-Object-Detection/coco/images/test/probe_43_png.rf.tUpUvAh3TE00AHrGX1kD.png')

# Converter para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Faixa de verde
lower_green = np.array([60, 40, 40])
upper_green = np.array([100, 255, 255])

# Máscara
mask = cv2.inRange(hsv, lower_green, upper_green)

# Operações morfológicas para reduzir ruído
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Encontrar contornos
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos grandes
min_area = 500  # ajuste conforme o tamanho da haste
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Ordenar por área (maior primeiro)
filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

# Processar apenas contornos filtrados
for cnt in filtered_contours:
    area = cv2.contourArea(cnt)

    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Desenhar o centro
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

        #--------------------------
        #  ENCONTRAR A PONTA CÔNICA
        #--------------------------
        pts = cnt.reshape(-1, 2)
        dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        idx = np.argmax(dists)
        cone_tip = tuple(pts[idx])

        # Desenhar a ponta do cone
        cv2.circle(img, cone_tip, 8, (255, 0, 0), -1)
        cv2.putText(img, "Cone", (cone_tip[0] + 10, cone_tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        #--------------------------
        #  CALCULAR ORIENTAÇÃO
        #--------------------------
        dx = cone_tip[0] - cx
        dy = cone_tip[1] - cy

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        # Escrever na imagem
        cv2.putText(img, f"{angle_deg:.1f} deg", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        print("Falha no cálculo do centro.")

# Mostrar resultado
cv2.imshow("Mascara", mask)
cv2.imshow("Resultado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()