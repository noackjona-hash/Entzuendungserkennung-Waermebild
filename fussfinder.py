import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V17).
    Nutzt intelligente Graustufen-Maskierung und geometrische Zehen-Reihen-Isolierung.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Bessere Maskierung: Das GESAMTE Fußprofil erfassen (auch kalte lila Zehen)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Otsu berechnen, aber Schwelle drastisch senken (35%), um lila/dunkelrote Ränder zu behalten
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = max(15, int(otsu_val * 0.35))
        _, thresh = cv2.threshold(blurred, low_thresh, 255, cv2.THRESH_BINARY)

        # Morphologische Bereinigung
        k_size = max(5, int(img_w * 0.01))
        kernel = np.ones((k_size, k_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Die 2 größten Konturen holen (die Füße)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        # Sortiere Füße im Bild von Links nach Rechts
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            # Rauschen ignorieren
            if cv2.contourArea(contour) < (img_w * img_h * 0.015):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 35% des Fußes)
            top_limit = y + int(h * 0.35) 
            oberkante = {}
            for pt in contour:
                px, py = pt[0]
                if py < top_limit:
                    if px not in oberkante or py < oberkante[px]:
                        oberkante[px] = py
                        
            if len(oberkante) < 20: 
                continue
                
            # 3. Zehen-Reihe exakt isolieren
            # Finde den absolut höchsten Punkt des Fußes (Minimum Y auf dem Bildschirm)
            min_y = min(oberkante.values())
            
            # Definiere ein horizontales Band für die Zehen (bis zu 20% der Fußhöhe nach unten)
            toe_band_limit = min_y + int(h * 0.20)
            
            toe_x_coords = [px for px, py in oberkante.items() if py <= toe_band_limit]
            
            if not toe_x_coords:
                continue
                
            span_min_x = min(toe_x_coords)
            span_max_x = max(toe_x_coords)
            
            # 4. In 5 exakte Spalten aufteilen
            segment_width = (span_max_x - span_min_x) / 5.0
            foot_points = []
            
            for i in range(5):
                seg_start = span_min_x + i * segment_width
                seg_end = span_min_x + (i + 1) * segment_width
                
                seg_x_coords = [px for px in oberkante.keys() if seg_start <= px <= seg_end]
                
                if seg_x_coords:
                    best_x = min(seg_x_coords, key=lambda px: oberkante[px])
                    best_y = oberkante[best_x]
                    
                    # V17: Punkt ca. 4% nach unten ins MTP Gelenk schieben
                    offset_y = int(h * 0.04)
                    
                    foot_points.append({
                        "x": best_x,
                        "y": best_y + offset_y
                    })
                    
            # 5. Korrekte Benennung der 5 Punkte
            for i, pt in enumerate(foot_points):
                if fuss_name == "Linker Fuß":
                    # Im Bild links -> Anatomisch linker Fuß -> Zeh 1 (groß) ist rechts
                    zeh_nr = 5 - i
                else:
                    # Im Bild rechts -> Anatomisch rechter Fuß -> Zeh 1 (groß) ist links
                    zeh_nr = i + 1
                    
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points