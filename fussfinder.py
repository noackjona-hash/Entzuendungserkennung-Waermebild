import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V15).
    Skaliert dynamisch mit der Bildgröße! Perfekt für verschiedene Kameras.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphologische Kernel relativ zur Bildbreite
        k_size = max(3, int(img_w * 0.005))
        k_large = max(5, int(img_w * 0.015))
        kernel_small = np.ones((k_size, k_size), np.uint8)
        kernel_large = np.ones((k_large, k_large), np.uint8)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: return detected_points
            
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
        
        for f_idx, contour in enumerate(sorted_contours):
            # Ignoriere winzige Artefakte (kleiner als 1% der Bildfläche)
            if cv2.contourArea(contour) < (img_w * img_h * 0.01):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            top_limit = y + int(h * 0.40) 
            oberkante = {}
            
            for pt in contour:
                px, py = pt[0]
                if py < top_limit:
                    if px not in oberkante or py < oberkante[px]:
                        oberkante[px] = py
                        
            if len(oberkante) < 15: continue
                
            sorted_x = sorted(oberkante.keys())
            raw_y = [oberkante[px] for px in sorted_x]
            
            window_size = max(5, int(w * 0.1)) # Fenstergröße abhängig von Fußbreite
            if len(raw_y) < window_size: continue
                
            kernel = np.ones(window_size) / window_size
            smoothed_y = np.convolve(raw_y, kernel, mode='valid')
            valid_x = sorted_x[window_size//2 : -window_size//2 + 1]
            
            peaks = []
            prominenz_threshold = max(2.0, h * 0.01) # Prominenz skaliert mit Fußhöhe
            
            for i in range(3, len(smoothed_y) - 3):
                if (smoothed_y[i] < smoothed_y[i-1] and smoothed_y[i] < smoothed_y[i-2] and 
                    smoothed_y[i] < smoothed_y[i+1] and smoothed_y[i] < smoothed_y[i+2]):
                    
                    val_left = max(smoothed_y[max(0, i-10):i]) if i > 0 else smoothed_y[i]
                    val_right = max(smoothed_y[i+1:min(len(smoothed_y), i+11)]) if i < len(smoothed_y)-1 else smoothed_y[i]
                    
                    prominenz = max(val_left - smoothed_y[i], val_right - smoothed_y[i])
                    
                    if prominenz > prominenz_threshold:  
                        px = valid_x[i]
                        py = oberkante[px]
                        # V15: Anker rutscht prozentual nach unten (ca. 5% der Fußhöhe) ins Gelenk
                        peaks.append((px, py + int(h * 0.05))) 

            # NMS - Punkte die zu nah sind filtern (Abstand skaliert mit Fußbreite)
            min_abstand = w * 0.1 
            filtered_peaks = []
            for p in peaks:
                too_close = False
                for fp in filtered_peaks:
                    if abs(p[0] - fp[0]) < min_abstand:
                        too_close = True
                        break
                if not too_close:
                    filtered_peaks.append(p)
                    
            for i, (px, py) in enumerate(filtered_peaks[:5]):
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {i+1}",
                    "punkt": (int(px), int(py))
                })
                
        return detected_points