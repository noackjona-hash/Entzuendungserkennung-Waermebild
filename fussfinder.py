import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V20).
    Nutzt 'Detrended Baseline Slicing', um die Fußschräge mathematisch zu neutralisieren.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung (Perfekt für FLIR Ironbow)
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
            
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.005)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 35%)
            top_limit = y + int(h * 0.35) 
            top_pts = [pt[0] for pt in contour if pt[0][1] < top_limit]
            
            if len(top_pts) < (w * 0.3): 
                continue
                
            oberkante = {}
            for px, py in top_pts:
                if px not in oberkante or py < oberkante[px]:
                    oberkante[px] = py
                    
            sorted_x = sorted(oberkante.keys())
            
            # 3. Ankerpunkt (Großer Zeh = Höchster Punkt im Bild) finden
            absolute_min_y = min(oberkante.values())
            peak_x_candidates = [px for px in sorted_x if oberkante[px] == absolute_min_y]
            peak_x = sum(peak_x_candidates) // len(peak_x_candidates)
            center_x = x + w / 2.0
            
            is_right_foot = peak_x < center_x
            
            # 4. Zehen-Zone und Detrending-Baseline definieren
            if is_right_foot:
                start_x = peak_x - int(w * 0.15)
                end_x = peak_x + int(w * 0.70) 
            else:
                start_x = peak_x - int(w * 0.70) 
                end_x = peak_x + int(w * 0.15)
                
            start_x = max(start_x, sorted_x[0])
            end_x = min(end_x, sorted_x[-1])
            
            toe_zone_x = [px for px in sorted_x if start_x <= px <= end_x]
            if not toe_zone_x: 
                continue
                
            # Die Baseline verbindet den großen Zeh mit dem Rand der kleinen Zehen
            p_big = (peak_x, absolute_min_y)
            p_small = (end_x, oberkante[end_x]) if is_right_foot else (start_x, oberkante[start_x])
            
            # Steigung m der Schräge berechnen
            if p_small[0] != p_big[0]:
                m = (p_small[1] - p_big[1]) / (p_small[0] - p_big[0])
            else:
                m = 0
                
            # Höhe über der Baseline berechnen (Neutralisierung der Schräge)
            detrended_heights = {}
            for px in toe_zone_x:
                py = oberkante[px]
                # Y-Wert auf der geraden Baseline an der Stelle X
                baseline_y = m * (px - p_big[0]) + p_big[1]
                # Wie weit sticht die Kontur nach oben heraus? (>0 = Zeh)
                detrended_heights[px] = baseline_y - py  
                
            # 5. In 5 Segmente schneiden und wahre Spitzen finden
            span_width = toe_zone_x[-1] - toe_zone_x[0]
            segment_width = span_width / 5.0
            foot_points = []
            
            for i in range(5):
                seg_start = toe_zone_x[0] + i * segment_width
                seg_end = toe_zone_x[0] + (i + 1) * segment_width
                
                seg_px = [px for px in toe_zone_x if seg_start <= px <= seg_end]
                
                if seg_px:
                    # Der Zeh ist der Punkt, der am weitesten ÜBER die Baseline hinausragt!
                    best_px = max(seg_px, key=lambda px: detrended_heights[px])
                    best_py = oberkante[best_px]
                    
                    # Punkt um ~4% der Fußhöhe nach unten ins Gelenk schieben
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"x": best_px, "y": best_py + offset_y})
                else:
                    # Fallback
                    fallback_x = int(seg_start + segment_width / 2)
                    fallback_y = int(m * (fallback_x - p_big[0]) + p_big[1])
                    foot_points.append({"x": fallback_x, "y": fallback_y})
                    
            # 6. Anatomisch korrekte Benennung
            for i, pt in enumerate(foot_points):
                zeh_nr = i + 1 if is_right_foot else 5 - i
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points