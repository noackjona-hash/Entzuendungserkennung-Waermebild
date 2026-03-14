import cv2
import numpy as np
import math

def analyze_extremities_and_smart_heat(image, gray_img):
    img_h, img_w = gray_img.shape
    
    # 1. Weichzeichnen, um Gradienten zu glätten (Profi-Schritt!)
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    # 2. Binarisieren (Vordergrund/Hintergrund)
    _, mask = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)

    # 3. Schwerpunkt
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    cv2.circle(image, (cx, cy), 6, (255, 0, 0), -1)

    # 4. Gummiband und obere Spitzen
    hull = cv2.convexHull(contour)
    merged_tips = []
    for point in hull:
        x, y = point[0]
        if y < cy - 20: 
            is_new = True
            for i, mt in enumerate(merged_tips):
                if math.sqrt((x-mt[0])**2 + (y-mt[1])**2) < 35:
                    is_new = False
                    if y < mt[1]: merged_tips[i] = (x, y)
                    break
            if is_new: merged_tips.append((x, y))

    merged_tips = sorted(merged_tips, key=lambda x: x[0])

    # 5. Smarte Entzündungs-Logik: Temperatur-Gradient
    inflamed_clusters = []
    for i, tip in enumerate(merged_tips):
        temp = blurred[tip[1], tip[0]]
        
        # Ein echtes Cluster ist heiß UND der Gradient (Abfall) ist hoch.
        # Wir messen die Hitze im Umkreis
        margin = 15
        if tip[0] > margin and tip[1] > margin and tip[0] < img_w - margin and tip[1] < img_h - margin:
            roi = blurred[tip[1]-margin : tip[1]+margin, tip[0]-margin : tip[0]+margin]
            local_avg = np.mean(roi)
            
            # Gradient = Maximale Hitze im Cluster minus lokale Durchschnitts-Hitze
            gradient = temp - local_avg
            
            # DIAGNOSE: Cluster ist heiß (>210) UND der Gradient ist stark (>15)
            if temp > 210 and gradient > 15:
                inflamed_clusters.append(tip)
                
        # Zeichnen der normalen Spitzen
        cv2.line(image, (cx, cy), tip, (255, 255, 255), 1)
        cv2.circle(image, tip, 5, (0, 255, 0), -1)

    # 6. Smarte Bounding Box um alle Entzündungs-Cluster zeichnen
    if inflamed_clusters:
        # Konvertieren in NumPy Array für Berechnungen
        pts = np.array(inflamed_clusters)
        
        # Die kleinste umschließende Box finden
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Die Box ist etwas zu eng, wir blähen sie auf
        infl_x, infl_y, infl_w, infl_h = cv2.boundingRect(pts)
        padding = 20
        start_pt = (infl_x - padding, infl_y - padding)
        end_pt = (infl_x + infl_w + padding, infl_y + infl_h + padding)
        
        # Box und Warn-Label zeichnen
        cv2.rectangle(image, start_pt, end_pt, (0, 0, 255), 3) # ROT
        cv2.putText(image, "WARNUNG: Lokaler Entzuendungs-Cluster", (start_pt[0], start_pt[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Den heißesten Punkt im Cluster finden
        gray_roi = gray_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
        _, max_val, _, _ = cv2.minMaxLoc(gray_roi)
        print(f"WARNUNG: Entzuendungs-Cluster erkannt! Spitzenwert: {max_val}")
    else:
        print("Normal: Keine lokalen Entzuendungs-Cluster erkannt.")

    return image