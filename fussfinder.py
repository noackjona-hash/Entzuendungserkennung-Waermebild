import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V18).
    Nutzt 'Dynamisches Slicing' für absolute Immunität gegen schräge/geneigte Füße.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung: Zuverlässig für FLIR-Bilder (Hintergrund ist extrem dunkel)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Ein fester Schwellenwert von 30 ist perfekt für FLIR Ironbow,
        # um den Fuß vom Hintergrund zu trennen, OHNE kalte lila Zehen abzuschneiden.
        _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        # Morphologische Bereinigung, um kleine Lücken zu schließen
        k_size = max(3, int(img_w * 0.008))
        kernel_open = np.ones((k_size, k_size), np.uint8)
        kernel_close = np.ones((k_size * 2, k_size * 2), np.uint8)
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Filtere kleine Störpixel heraus und nimm die beiden größten Flächen (Füße)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.01)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Sortiere die Füße im Bild von Links nach Rechts
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 35% des Bounding Boxes)
            top_limit = y + int(h * 0.35) 
            oberkante = {}
            for pt in contour:
                px, py = pt[0]
                if py < top_limit:
                    if px not in oberkante or py < oberkante[px]:
                        oberkante[px] = py
                        
            # Wenn die Kontur oben zu schmal ist, überspringen
            if len(oberkante) < (w * 0.3): 
                continue
                
            # 3. Den Ankerpunkt (Großer Zeh = Höchster Punkt / Minimum Y) finden
            absolute_min_y = min(oberkante.values())
            peak_x_candidates = [px for px, py in oberkante.items() if py == absolute_min_y]
            peak_x = sum(peak_x_candidates) // len(peak_x_candidates)
            
            center_x = x + w / 2
            
            # 4. Dynamisches Slicing der Zehen-Region
            # Wir definieren den Bereich, in dem sich die Zehen befinden, 
            # ausgehend von der Position des großen Zehs.
            if peak_x < center_x:
                # Rechter Fuß (Großer Zeh ist links auf dem Bildschirm)
                start_x = peak_x - int(w * 0.05)
                end_x = peak_x + int(w * 0.75) # 75% der Breite nach rechts für die kleinen Zehen
            else:
                # Linker Fuß (Großer Zeh ist rechts auf dem Bildschirm)
                start_x = peak_x - int(w * 0.75) # 75% der Breite nach links für die kleinen Zehen
                end_x = peak_x + int(w * 0.05)
                
            # Grenzen absichern
            min_contour_x = min(oberkante.keys())
            max_contour_x = max(oberkante.keys())
            start_x = max(start_x, min_contour_x)
            end_x = min(end_x, max_contour_x)
            
            # 5. In 5 exakte Spalten unterteilen und in jeder den höchsten Punkt suchen
            segment_width = (end_x - start_x) / 5.0
            foot_points = []
            
            for i in range(5):
                seg_start = start_x + i * segment_width
                seg_end = start_x + (i + 1) * segment_width
                
                # Punkte, die in diese Spalte fallen
                seg_pts = [(px, py) for px, py in oberkante.items() if seg_start <= px <= seg_end]
                
                if seg_pts:
                    # Der höchste Punkt (min Y) in dieser Spalte ist der Zeh!
                    best_px, best_py = min(seg_pts, key=lambda item: item[1])
                    
                    # Punkt um ~4% der Fußhöhe nach unten in das Gelenk verschieben
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"x": best_px, "y": best_py + offset_y})
                else:
                    # Fallback, falls eine Spalte leer ist (z.B. Lücke zwischen Zehen)
                    fallback_x = int(seg_start + segment_width / 2)
                    foot_points.append({"x": fallback_x, "y": absolute_min_y + int(h * 0.08)})
                    
            # 6. Anatomisch korrekte Benennung der 5 Punkte
            for i, pt in enumerate(foot_points):
                # Die Liste foot_points ist immer von links nach rechts geordnet
                if peak_x < center_x:
                    # Großer Zeh ist links (Index 0)
                    zeh_nr = i + 1
                else:
                    # Großer Zeh ist rechts (Index 4)
                    zeh_nr = 5 - i
                    
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points