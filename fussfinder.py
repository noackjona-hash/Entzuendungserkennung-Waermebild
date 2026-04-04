import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V19).
    Nutzt 'Radial Sector Slicing' für perfekte Spitzen-Detektion bei schrägen Füßen.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung (Zuverlässig für FLIR Ironbow)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        k_size = max(3, int(img_w * 0.008))
        kernel_open = np.ones((k_size, k_size), np.uint8)
        kernel_close = np.ones((k_size * 2, k_size * 2), np.uint8)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Toleranz auf 0.5% gesenkt, um auch kältere (kleinere) Füße sicher zu erfassen
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.005)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren
            top_limit = y + int(h * 0.35) 
            oberkante_pts = [pt[0] for pt in contour if pt[0][1] < top_limit]
            
            if len(oberkante_pts) < (w * 0.3): 
                continue
                
            # 3. Den Großen Zeh (absoluten Hochpunkt) finden, um die Fußseite abzuschneiden
            absolute_min_y = min([p[1] for p in oberkante_pts])
            peak_x_candidates = [p[0] for p in oberkante_pts if p[1] == absolute_min_y]
            peak_x = sum(peak_x_candidates) // len(peak_x_candidates)
            center_x = x + w / 2.0
            
            if peak_x < center_x: # Rechter Fuß (Großer Zeh ist links)
                start_x = peak_x - int(w * 0.10)
                end_x = peak_x + int(w * 0.75) 
            else: # Linker Fuß (Großer Zeh ist rechts)
                start_x = peak_x - int(w * 0.75) 
                end_x = peak_x + int(w * 0.10)
                
            start_x = max(start_x, min([p[0] for p in oberkante_pts]))
            end_x = min(end_x, max([p[0] for p in oberkante_pts]))
            
            # Die saubere Zehen-Reihe
            toe_pts = [p for p in oberkante_pts if start_x <= p[0] <= end_x]
            if not toe_pts: 
                continue
                
            # 4. Radiales Slicing Zentrum (Der "Pivot-Punkt" im Mittelfuß)
            cx = (start_x + end_x) / 2.0
            cy = y + h * 0.45 
            
            # Umwandlung in Polarkoordinaten
            polar_pts = []
            for px, py in toe_pts:
                dx = px - cx
                dy = cy - py # Invertiert, Y wächst nach oben
                if dy > 0:
                    theta = np.degrees(np.arctan2(dy, dx)) # Winkel
                    r = np.hypot(dx, dy) # Distanz zum Zentrum
                    polar_pts.append((theta, r, px, py))
                    
            if not polar_pts: 
                continue
                
            polar_pts.sort(key=lambda item: item[0])
            min_theta = polar_pts[0][0]
            max_theta = polar_pts[-1][0]
            
            # 5. In 5 fächerförmige Winkelsektoren unterteilen
            theta_range = max_theta - min_theta
            sector_size = theta_range / 5.0
            foot_points = []
            
            for i in range(5):
                sec_min = min_theta + i * sector_size
                sec_max = min_theta + (i + 1) * sector_size
                
                sec_pts = [p for p in polar_pts if sec_min <= p[0] <= sec_max]
                
                if sec_pts:
                    # MAXIMALER RADIUS (r): Findet die physisch abstehende Zehenspitze!
                    best_pt = max(sec_pts, key=lambda p: p[1])
                    
                    # Punkt um ~4% der Fußhöhe nach unten in das Gelenk verschieben
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"px": best_pt[2], "py": best_pt[3] + offset_y})
                else:
                    # Fallback Interpolation, falls ein Sektor komplett leer ist
                    fallback_theta = sec_min + sector_size / 2
                    fallback_px = int(cx + np.cos(np.radians(fallback_theta)) * (h * 0.2))
                    fallback_py = int(cy - np.sin(np.radians(fallback_theta)) * (h * 0.2))
                    foot_points.append({"px": fallback_px, "py": fallback_py})
                    
            # 6. Anatomisch korrekte Benennung (Sektoren laufen von Rechts nach Links)
            for i, pt in enumerate(foot_points):
                if peak_x < center_x:
                    zeh_nr = 5 - i # Rechter Fuß: Großer Zeh (1) ist links (Index 4)
                else:
                    zeh_nr = i + 1 # Linker Fuß: Großer Zeh (1) ist rechts (Index 0)
                    
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['px']), int(pt['py']))
                })
                
        return detected_points