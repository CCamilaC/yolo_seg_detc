import pyrealsense2 as rs
import numpy as np
import cv2
import math

# Configurar câmera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Parâmetros de detecção de verde
lower_green = np.array([50, 120, 80])
upper_green = np.array([85, 255, 255])
kernel = np.ones((5,5), np.uint8)
min_area = 500  # ajuste conforme o tamanho da haste

# -------------------------------
# Funções auxiliares
# -------------------------------
def get_orientation(contour):
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90
    return angle

def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]])

def merge_if_same_orientation(contours, angle_threshold=12, max_distance=200):
    """
    Une contornos separados que possuem a mesma orientação e estão próximos.
    """
    if len(contours) <= 1:
        return contours

    merged = []
    used = [False] * len(contours)
    centers = [get_center(c) for c in contours]

    for i in range(len(contours)):
        if used[i]:
            continue
        base_angle = get_orientation(contours[i])
        center_i = centers[i]
        if center_i is None:
            continue
        group = [contours[i]]
        used[i] = True

        for j in range(i+1, len(contours)):
            if used[j]:
                continue
            angle_j = get_orientation(contours[j])
            center_j = centers[j]
            if center_j is None:
                continue
            distance = np.linalg.norm(center_i - center_j)
            if abs(base_angle - angle_j) < angle_threshold and distance <= max_distance:
                group.append(contours[j])
                used[j] = True

        merged_contour = np.vstack(group)
        merged.append(merged_contour)

    return merged

# -------------------------------
# Loop principal
# -------------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # -----------------------------
        # Canny + bounding box (unido)
        # -----------------------------
        edges = cv2.Canny(mask, 50, 150)
        contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_canny:
            contours_canny = merge_if_same_orientation(contours_canny)
            for c in contours_canny:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(output, "Probe", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # -----------------------------
        # Contornos HSV (unidos)
        # -----------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        filtered_contours = merge_if_same_orientation(filtered_contours)

        coords = []
        for cnt in filtered_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                coords.append((cx, cy))
                cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

                pts = cnt.reshape(-1, 2)
                dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
                idx = np.argmax(dists)
                cone_tip = tuple(pts[idx])
                cv2.circle(output, cone_tip, 8, (255, 0, 0), -1)
                cv2.putText(output, "Cone", (cone_tip[0] + 10, cone_tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.line(output, (cx, cy), cone_tip, (0, 255, 0), 2)

                dx = cone_tip[0] - cx
                dy = cone_tip[1] - cy
                angle_rad = math.atan2(dy, dx)
                angle_deg = (math.degrees(angle_rad) + 360) % 360
                cv2.putText(output, f"{angle_deg:.1f} deg", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # -----------------------------
        # Impressão formatada no terminal
        # -----------------------------
        print("\nNumber of detected objects:", len(coords))
        print("Position in pixels of the objects:")
        if coords:
            for i, (cx, cy) in enumerate(coords):
                print(f"      Object {i+1}: x = {cx} ; y = {cy}")
        else:
            print("      None")

        cv2.imshow("Mask", mask)
        cv2.imshow("Deteccao Cone", output)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()