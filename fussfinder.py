import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V21).
    Nutzt dynamisches Otsu-Masking und 'Polyfit Detrending' für absolute Präzision.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung: Dynamischer Otsu (extrem niedriger Cutoff für kalte Zehen)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Otsu analysiert das Bild. Wir nehmen 25% des Otsu-Wertes, 
        # um sicherzustellen, dass auch tief-lila Zehen im Ironbow-Schema erfasst werden!
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = max(10, int(otsu_val * 0.25))
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphologie: Lücken schließen und Rauschen entfernen
        k_size = max(3, int(img_w * 0.01))
        kernel_open = np.ones((k_size, k_size), np.uint8)
        kernel_close = np.ones((k_size * 2, k_size * 2), np.uint8)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Die beiden größten Konturen (die Füße) filtern
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.005)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Von Links nach Rechts im Bild sortieren
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 35%)
            top_limit = y + int(h * 0.35) 
            top_pts = [pt[0] for pt in contour if pt[0][1] < top_limit]
            
            # Abbruch, wenn die Oberkante zu wenig Pixel hat
            if len(top_pts) < (w * 0.2): 
                continue
                
            oberkante = {}
            for px, py in top_pts:
                if px not in oberkante or py < oberkante[px]:
                    oberkante[px] = py
                    
            sorted_x = sorted(oberkante.keys())
            edge_y = [oberkante[px] for px in sorted_x]
            
            # 3. Mathematische Detrending-Linie (Polyfit Regression)
            # Berechnet die generelle Schräglage des Fußes viel genauer als nur 2 Endpunkte
            try:
                m, b = np.polyfit(sorted_x, edge_y, 1)
            except np.linalg.LinAlgError:
                m, b = 0, y # Fallback bei extremen Rechenfehlern
                
            # 4. In 5 exakte Spalten aufteilen
            span_width = sorted_x[-1] - sorted_x[0]
            segment_width = span_width / 5.0
            foot_points = []
            
            for i in range(5):
                seg_start = sorted_x[0] + i * segment_width
                seg_end = sorted_x[0] + (i + 1) * segment_width
                
                seg_px = [px for px in sorted_x if seg_start <= px <= seg_end]
                
                if seg_px:
                    # Die Zehenspitze ragt am weitesten über die Trendlinie hinaus!
                    # Trendlinie_Y = m * px + b
                    # Distanz nach "oben" = Trendlinie_Y - Tatsächliches_Y
                    best_px = max(seg_px, key=lambda px: (m * px + b) - oberkante[px])
                    best_py = oberkante[best_px]
                    
                    # Punkt ca. 4% der Fußhöhe nach unten ins Gelenk verschieben
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"x": best_px, "y": best_py + offset_y})
                else:
                    # Fallback für eine leere Spalte
                    fallback_x = int(seg_start + segment_width / 2)
                    fallback_y = int(m * fallback_x + b)
                    foot_points.append({"x": fallback_x, "y": fallback_y})
                    
            # 5. Anatomisch korrekte Benennung
            # Linker Fuß im Bild -> Anatomisch Linker Fuß -> Großer Zeh ist RECHTS
            # Rechter Fuß im Bild -> Anatomisch Rechter Fuß -> Großer Zeh ist LINKS
            for i, pt in enumerate(foot_points):
                if fuss_name == "Linker Fuß":
                    zeh_nr = 5 - i # Läuft von 5 nach 1
                else:
                    zeh_nr = i + 1 # Läuft von 1 nach 5
                    
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points