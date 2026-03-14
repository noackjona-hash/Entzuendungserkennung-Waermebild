import cv2
import numpy as np
import math

def analyze_extremities_and_smart_heat(image, gray_img):
    _, mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    cv2.circle(image, (cx, cy), 6, (255, 0, 0), -1)

    hull = cv2.convexHull(contour)
    raw_tips = []
    for point in hull:
        x, y = point[0]
        if y < cy - 20: 
            raw_tips.append((x, y))

    merged_tips = []
    for tip in raw_tips:
        is_new = True
        for i, mt in enumerate(merged_tips):
            if math.sqrt((tip[0]-mt[0])**2 + (tip[1]-mt[1])**2) < 35:
                is_new = False
                if tip[1] < mt[1]:
                    merged_tips[i] = tip
                break
        if is_new:
            merged_tips.append(tip)

    # BUGFIX 1: Top-5-Regel. Wir nehmen zwingend nur die 5 Punkte, die am hoechsten liegen (kleinstes Y)
    # Damit filtern wir die komplette Seite des Fusses heraus!
    merged_tips = sorted(merged_tips, key=lambda x: x[1])[:5]
    
    # Danach wieder von links nach rechts sortieren
    merged_tips = sorted(merged_tips, key=lambda x: x[0])

    tips_with_temp = []
    for tip in merged_tips:
        x, y = tip
        # BUGFIX 2: Der "Fleisch-Sensor". 
        # Anstatt genau auf der Aussenkante zu messen, ziehen wir eine Box leicht nach unten (in den Zeh hinein)
        # und nehmen davon den maximalen Helligkeitswert.
        roi = gray_img[y : min(gray_img.shape[0], y + 15), max(0, x - 10) : min(gray_img.shape[1], x + 10)]
        
        # Sicherstellen, dass die ROI nicht leer ist
        if roi.size > 0:
            temp = int(np.max(roi))
        else:
            temp = 0
            
        tips_with_temp.append((tip, temp))
        
    if not tips_with_temp:
        return image

    avg_toe_temp = sum(t[1] for t in tips_with_temp) / len(tips_with_temp)
    
    cv2.putText(image, f"Zehen-Durchschnitt: {int(avg_toe_temp)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for tip, temp in tips_with_temp:
        # LOGIK: Ist dieser Zeh signifikant heisser? 
        # (Da wir jetzt tief im warmen Zeh messen, ist +15 wieder ein super Wert)
        is_inflamed = temp > avg_toe_temp + 15 
        
        if is_inflamed:
            box_width = 30
            box_height_up = 20
            box_height_down = 60
            
            start_pt = (tip[0] - box_width, tip[1] - box_height_up)
            end_pt = (tip[0] + box_width, tip[1] + box_height_down)
            
            cv2.rectangle(image, start_pt, end_pt, (0, 0, 255), 2)
            cv2.drawMarker(image, tip, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(image, f"HOT: {temp}", (start_pt[0], start_pt[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.circle(image, tip, 5, (0, 255, 0), -1)
            cv2.putText(image, str(temp), (tip[0] - 15, tip[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return image