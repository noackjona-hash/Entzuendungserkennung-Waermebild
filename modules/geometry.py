import cv2
import numpy as np

def find_both_feet(gray_img):
    # 1. Starke Rauschunterdrueckung fuer perfekte Masken
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)
    
    # 2. ADAPTIVE OTSU-BINARISIERUNG (Wissenschaftlicher Standard)
    # Berechnet den optimalen Schwellenwert selbst, statt "40" hart zu codieren!
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Morphologisches "Bügeln" (schliesst Loecher in der Maske)
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        return None, None, None
        
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    sorted_by_x = sorted(sorted_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    return sorted_by_x[0], sorted_by_x[1], mask

def extract_toes_from_contour(contour, gray_img, foot_mask):
    M = cv2.moments(contour)
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # 1. Alle Punkte extrahieren, die in der oberen Haelfte des Fusses liegen
    upper_points = [pt[0] for pt in contour if pt[0][1] < cy - 10]
    
    if len(upper_points) < 5:
        return []

    # Formatieren fuer die OpenCV K-Means KI
    points_array = np.array(upper_points, dtype=np.float32)

    # 2. MACHINE LEARNING: K-Means Clustering
    # Wir zwingen die KI, die obere Fusskante in exakt 5 Regionen (Zehen) zu unterteilen!
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, _, centers = cv2.kmeans(points_array, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 3. Die 5 gefundenen Cluster-Zentren von links nach rechts sortieren
    centers = sorted(centers, key=lambda c: c[0])

    tips_with_data = []
    for center in centers:
        x, y = int(center[0]), int(center[1])
        
        # 4. Deep-Sensor: Ausgehend vom Cluster-Zentrum den absoluten Hitze-Pixel suchen
        roi_size = 40
        x_start, x_end = max(0, x - roi_size), min(gray_img.shape[1], x + roi_size)
        y_start, y_end = max(0, y - 15), min(gray_img.shape[0], y + 60) # Tiefer in den Zeh schauen
        
        roi_gray = gray_img[y_start:y_end, x_start:x_end]
        roi_mask = foot_mask[y_start:y_end, x_start:x_end] 
        
        if roi_gray.size > 0:
            # Maskierte Suche: Wir suchen den Hotspot AUSSCHLIESSLICH im Fussgewebe
            _, max_val, _, max_loc_roi = cv2.minMaxLoc(roi_gray, mask=roi_mask)
            temp = int(max_val)
            meas_pt = (x_start + max_loc_roi[0], y_start + max_loc_roi[1])
        else:
            temp, meas_pt = 0, (x, y)
            
        tips_with_data.append({"tip": (x, y), "temp": temp, "sensor": meas_pt})
        
    return tips_with_data