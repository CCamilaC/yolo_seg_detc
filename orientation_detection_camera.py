import pyrealsense2 as rs
import numpy as np
import cv2
import math

# Configurar câmera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Iniciar câmera
pipeline.start(config)

# Parâmetros de detecção de verde
lower_green = np.array([50, 70, 40])
upper_green = np.array([85, 255, 255])
kernel = np.ones((5,5), np.uint8)
min_area = 500  # ajuste conforme o tamanho da haste

try:
    while True:
        # Capturar frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Converter para array numpy
        frame = np.asanyarray(color_frame.get_data())
        output = frame.copy()

        # Converter para HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Criar máscara
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

        # Processar cada contorno
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

                # Encontrar ponta cônica
                pts = cnt.reshape(-1, 2)
                dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                idx = np.argmax(dists)
                cone_tip = tuple(pts[idx])
                cv2.circle(output, cone_tip, 8, (255, 0, 0), -1)
                cv2.putText(output, "Cone", (cone_tip[0] + 10, cone_tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


                cv2.line(output, (cx, cy), cone_tip, (0, 255, 0), 2)  # linha verde

                # Calcular orientação
                dx = cone_tip[0] - cx
                dy = cone_tip[1] - cy
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360
                cv2.putText(output, f"{angle_deg:.1f} deg", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mostrar resultado
        cv2.imshow("Mask", mask)
        cv2.imshow("Deteccao Cone", output)

        # Teclas
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()