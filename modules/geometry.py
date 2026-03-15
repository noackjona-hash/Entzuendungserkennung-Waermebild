import cv2
import numpy as np
import math
from scipy.signal import find_peaks

def find_both_feet(gray_img):
    # 1. Rauschunterdrueckung (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)
    
    # 2. Adaptive Binarisierung (besser bei ungleicher Ausleuchtung)
    _, mask = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    
    # 3. Morphologische Operationen: Loecher schliessen und Rauschen entfernen
    kernel = np.ones((7,7), np.uint8)
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
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # Nur die obere Haelfte der Fuss-Kontur betrachten
    upper_contour = [pt[0] for pt in contour if pt[0][1] < cy]
    
    if not upper_contour:
        return []

    # Distanz vom Zentrum zu jedem Punkt berechnen (Signal-Profil erstellen)
    distances = [math.sqrt((pt[0]-cx)**2 + (pt[1]-cy)**2) for pt in upper_contour]
    
    # SciPy Peak-Detection: Finde die 5 prominentesten Gipfel im Distanz-Signal
    # prominence=15 sorgt dafuer, dass kleine Dellen am Fussrand ignoriert werden
    peaks, properties = find_peaks(distances, prominence=15, distance=30)
    
    # Die 5 stärksten Peaks auswählen
    if len(peaks) > 5:
        prominences = properties['prominences']
        top_peak_indices = np.argsort(prominences)[-5:]
        peaks = [peaks[i] for i in top_peak_indices]
        
    raw_tips = [upper_contour[p] for p in peaks]
    
    # Von Links nach Rechts sortieren
    raw_tips = sorted(raw_tips, key=lambda x: x[0]) 

    tips_with_data = []
    for tip in raw_tips:
        x, y = tip
        x_start, x_end = max(0, x - 35), min(gray_img.shape[1], x + 35)
        y_start, y_end = max(0, y - 15), min(gray_img.shape[0], y + 65)
        
        roi_gray = gray_img[y_start:y_end, x_start:x_end]
        roi_mask = foot_mask[y_start:y_end, x_start:x_end] 
        
        if roi_gray.size > 0:
            # Suchen des absolut heissesten Pixels NUR innerhalb des Fusses (Maske)
            _, max_val, _, max_loc_roi = cv2.minMaxLoc(roi_gray, mask=roi_mask)
            temp = int(max_val)
            meas_pt = (x_start + max_loc_roi[0], y_start + max_loc_roi[1])
        else:
            temp, meas_pt = 0, (x, y)
            
        tips_with_data.append({"tip": tip, "temp": temp, "sensor": meas_pt})
        
    return tips_with_data