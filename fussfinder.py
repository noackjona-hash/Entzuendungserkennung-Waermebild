import cv2
import numpy as np
import math
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V24).
    Nutzt 'Geometric Rotation Alignment', um Fußschrägen vor dem Slicing mathematisch aufzulösen.
    Das ultimative und stabilste Modell für Wärmebilder.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung: Red-Channel + Dynamic Otsu Fusion
        # Sehr robust für FLIR Ironbow, schneidet kalte Zehen nicht ab
        _, _, r_channel = cv2.split(image)
        blurred = cv2.GaussianBlur(r_channel, (11, 11), 0)
        
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = max(15, int(otsu_val * 0.75)) 
        _, thresh = cv2.threshold(blurred, final_thresh, 255, cv2.THRESH_BINARY)

        # Morphologie zur Glättung der Ränder
        k_size = max(5, int(img_w * 0.01))
        kernel = np.ones((k_size, k_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Filtere Rauschen und hole die zwei größten Konturen (Füße)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.01)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 35%)
            top_limit = y + int(h * 0.35) 
            top_pts = [pt[0] for pt in contour if pt[0][1] < top_limit]
            
            if len(top_pts) < (w * 0.2): 
                continue
                
            # Erstelle ein sauberes Profil: Nur der höchste Y-Wert pro X-Spalte
            oberkante = {}
            for px, py in top_pts:
                if px not in oberkante or py < oberkante[px]:
                    oberkante[px] = py
                    
            sorted_x = sorted(oberkante.keys())
            
            # 3. Ankerpunkte für die Rotations-Achse finden
            min_y = min(oberkante.values())
            peak_x_candidates = [px for px in sorted_x if oberkante[px] == min_y]
            p_anchor_x = sum(peak_x_candidates) // len(peak_x_candidates)
            p_anchor_y = min_y
            
            center_x = x + w / 2.0
            is_right_foot = p_anchor_x < center_x
            
            # Finde den Rand der Zehenreihe (~75% der Fußbreite vom großen Zeh entfernt)
            if is_right_foot:
                target_edge_x = min(sorted_x[-1], p_anchor_x + int(w * 0.75))
            else:
                target_edge_x = max(sorted_x[0], p_anchor_x - int(w * 0.75))
                
            p_edge_x = min(sorted_x, key=lambda val: abs(val - target_edge_x))
            p_edge_y = oberkante[p_edge_x]
            
            # Berechne den Winkel (Theta) der Fußschräge
            dx = p_edge_x - p_anchor_x
            dy = p_edge_y - p_anchor_y
            theta = math.atan2(dy, dx)
            
            # Hilfsfunktion für die 2D-Rotation
            def rotate_point(px, py, cx, cy, angle):
                nx = math.cos(angle) * (px - cx) - math.sin(angle) * (py - cy) + cx
                ny = math.sin(angle) * (px - cx) + math.cos(angle) * (py - cy) + cy
                return nx, ny

            # 4. Fuß-Oberkante virtuell FLACH rotieren
            rotated_contour = []
            for px in sorted_x:
                py = oberkante[px]
                # Drehe um -Theta, um die Schräge exakt auszugleichen
                rx, ry = rotate_point(px, py, p_anchor_x, p_anchor_y, -theta)
                rotated_contour.append((rx, ry, px, py))
                
            # Nach dem neuen (flachen) X sortieren
            rotated_contour.sort(key=lambda p: p[0])
            
            # Die Grenzen der flachen Zehenreihe ermitteln
            rx_start = rotated_contour[0][0]
            rx_end = rotated_contour[-1][0]
            
            span_width = rx_end - rx_start
            if span_width <= 0: continue
            segment_width = span_width / 5.0
            
            foot_points = []
            
            # 5. In der flachen Ebene slicen und die Spitzen (Zehkuppen) greifen
            for i in range(5):
                seg_start = rx_start + i * segment_width
                seg_end = rx_start + (i + 1) * segment_width
                
                seg_pts = [p for p in rotated_contour if seg_start <= p[0] <= seg_end]
                
                if seg_pts:
                    # In der flachen Rotation ist das Minimum Y garantiert der perfekte Zeh!
                    best_pt = min(seg_pts, key=lambda p: p[1])
                    # Hole die ORIGINALEN Koordinaten dieses Punktes zurück
                    orig_x = best_pt[2]
                    orig_y = best_pt[3]
                    
                    # 4% Offset nach unten ins Gelenk (MTP)
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"x": orig_x, "y": orig_y + offset_y})
                else:
                    # Sicherer Fallback (Interpolation im flachen Raum -> zurück rotieren)
                    fallback_rx = seg_start + segment_width / 2.0
                    orig_fx, orig_fy = rotate_point(fallback_rx, p_anchor_y, p_anchor_x, p_anchor_y, theta)
                    foot_points.append({"x": orig_fx, "y": orig_fy + max(5, int(h * 0.04))})
                    
            # 6. Anatomisch korrekt durchnummerieren
            for i, pt in enumerate(foot_points):
                # foot_points läuft immer von links nach rechts
                # Rechter Fuß: Großer Zeh ist links (Index 0 = Zeh 1)
                # Linker Fuß: Großer Zeh ist rechts (Index 4 = Zeh 1)
                zeh_nr = i + 1 if is_right_foot else 5 - i
                
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points