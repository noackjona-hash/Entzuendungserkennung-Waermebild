import cv2
import numpy as np
import math

def analyze_shape_and_draw_skeleton(image, gray_img):
    # Bildhöhe und -breite auslesen, um die Ränder zu kennen
    img_h, img_w = gray_img.shape
    
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, "Unbekannt"

    contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except:
        defects = None

    tips = []

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            if b * c != 0: 
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / math.pi
            else:
                angle = 180
            
            if angle <= 90:
                if start[1] < cy + 80: tips.append(start)
                if end[1] < cy + 80: tips.append(end)

    # 1. NEUER FILTER: Rand-Rejektion (Border Rejection)
    # Ignoriert Punkte, die extrem nah am Rand des Fotos sind (Messfehler)
    margin = 20
    tips_inside = []
    for tip in tips:
        if tip[0] > margin and tip[0] < (img_w - margin) and tip[1] > margin and tip[1] < (img_h - margin):
            tips_inside.append(tip)

    # Distanz-Filter (doppelte Punkte zusammenfassen)
    filtered_tips = []
    for tip in tips_inside:
        is_new = True
        for ft in filtered_tips:
            if math.sqrt((tip[0]-ft[0])**2 + (tip[1]-ft[1])**2) < 40:
                is_new = False
                break
        if is_new:
            filtered_tips.append(tip)

    # 2. NEUER FILTER: Top 5 Prinzip
    # Wir sortieren die Punkte nach Y-Koordinate (Höhe). Das kleinste Y ist GANZ OBEN im Bild.
    # Wir nehmen nur die obersten 5 Punkte.
    filtered_tips = sorted(filtered_tips, key=lambda x: x[1])[:5]
    
    # Zum Schluss sortieren wir sie wieder von links nach rechts, das brauchen wir gleich
    filtered_tips = sorted(filtered_tips, key=lambda x: x[0])

    # Skelett einzeichnen
    for tip in filtered_tips:
        cv2.line(image, (cx, cy), tip, (0, 255, 255), 2)
        cv2.circle(image, tip, 6, (0, 0, 255), -1)

    if len(filtered_tips) >= 4:
        erkennung = "Hand"
    else:
        erkennung = "Fuss"

    return image, erkennung