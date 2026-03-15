import cv2
import numpy as np

def find_both_feet(gray_img):
    """Findet die Konturen der Füße mit adaptiver Otsu-Segmentierung."""
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None, None
        
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    sorted_by_x = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return sorted_by_x[0], sorted_by_x[1], mask

def extract_toes_with_ai(contour, gray_img, foot_mask):
    """Thermische Topologie: Nutzt K-Means Clustering direkt auf der Hitze-Verteilung."""
    x, y, w, h = cv2.boundingRect(contour)
    
    # 1. ORIENTIERUNGS-ERKENNUNG (Zehen oben oder unten?)
    # Wir vergleichen die Anzahl der Pixel (Masse) im oberen und unteren Fünftel
    mask_top = foot_mask[y:y+int(h*0.2), x:x+w]
    mask_bottom = foot_mask[y+int(h*0.8):y+h, x:x+w]
    
    if cv2.countNonZero(mask_top) > cv2.countNonZero(mask_bottom):
        # Zehen sind OBEN
        toe_y_start = y
        toe_y_end = y + int(h * 0.45) # Oberste 45% des Fusses
    else:
        # Zehen sind UNTEN (wie auf deinem 4. Bild!)
        toe_y_start = y + int(h * 0.55) # Unterste 45%
        toe_y_end = y + h

    # 2. ROI (Region of Interest) isolieren
    toe_mask = np.zeros_like(foot_mask)
    cv2.rectangle(toe_mask, (x, toe_y_start), (x+w, toe_y_end), 255, -1)
    
    # Nur Pixel behalten, die im ROI UND im echten Fuss sind
    final_toe_mask = cv2.bitwise_and(foot_mask, toe_mask)
    
    # 3. DIE HEISSESTEN PIXEL FILTERN (Top 30% der Hitze)
    roi_gray = cv2.bitwise_and(gray_img, gray_img, mask=final_toe_mask)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_img, mask=final_toe_mask)
    
    # Dynamischer Schwellenwert fuer die echten Hitze-Zentren
    hot_threshold = max_val - ((max_val - min_val) * 0.35)
    _, hot_mask = cv2.threshold(roi_gray, hot_threshold, 255, cv2.THRESH_BINARY)
    
    # Koordinaten aller heissen Pixel holen
    hot_points = cv2.findNonZero(hot_mask)
    
    tips_with_data = []
    
    if hot_points is not None and len(hot_points) >= 5:
        # 4. K-MEANS MACHINE LEARNING AUF DER HITZE-WOLKE
        points_array = np.float32(hot_points).reshape(-1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        
        # Die KI teilt die Hitzewolke mathematisch in exakt 5 Zentren!
        _, _, centers = cv2.kmeans(points_array, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Von Links nach Rechts sortieren
        centers = sorted(centers, key=lambda c: c[0])
        
        # 5. Praezisions-Sensor (Deep Sensor)
        for center in centers:
            cx, cy = int(center[0]), int(center[1])
            
            # Kleines Suchfenster um das KI-Zentrum für exakten Maximalwert
            r = 15
            x_s, x_e = max(0, cx-r), min(gray_img.shape[1], cx+r)
            y_s, y_e = max(0, cy-r), min(gray_img.shape[0], cy+r)
            
            local_gray = gray_img[y_s:y_e, x_s:x_e]
            local_mask = foot_mask[y_s:y_e, x_s:x_e]
            
            if local_gray.size > 0:
                _, t_max, _, max_loc = cv2.minMaxLoc(local_gray, mask=local_mask)
                temp = int(t_max)
                meas_pt = (x_s + max_loc[0], y_s + max_loc[1])
            else:
                temp, meas_pt = int(gray_img[cy, cx]), (cx, cy)
                
            tips_with_data.append({"tip": (cx, cy), "temp": temp, "sensor": meas_pt})
            
    else:
        print("Fehler: Konnte keine ausreichende Hitze-Signatur im Zehenbereich finden.")
        
    return tips_with_data