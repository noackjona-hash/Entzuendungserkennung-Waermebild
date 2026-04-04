import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V16).
    Kombiniert die Robustheit der Zonen-Aufteilung (V13) mit dynamischer Skalierung.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Der bewährte Roter-Kanal-Trick: 
        # FLIR Wärmebilder haben bei Hitze extrem hohe Rot-Werte (Gelb/Weiß). 
        # Das isoliert den Fuß perfekt vom dunklen/blauen Hintergrund.
        _, _, r_channel = cv2.split(image)
        blurred = cv2.GaussianBlur(r_channel, (11, 11), 0)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Schwelle leicht senken (80%), um auch kühlere Zehenspitzen zu erwischen
        _, thresh = cv2.threshold(blurred, int(otsu_val * 0.8), 255, cv2.THRESH_BINARY)

        # 2. Morphologische Bereinigung (skaliert mit Bildgröße)
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
            # Rauschen ignorieren (kleiner als 1.5% der Bildfläche)
            if cv2.contourArea(contour) < (img_w * img_h * 0.015):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Nur das obere Drittel analysieren (Zehenzone)
            top_limit = y + int(h * 0.35) 
            
            # Profil der Oberkante erstellen
            oberkante = {}
            for pt in contour:
                px, py = pt[0]
                if py < top_limit:
                    if px not in oberkante or py < oberkante[px]:
                        oberkante[px] = py
                        
            if len(oberkante) < 20: 
                continue
                
            sorted_x = sorted(oberkante.keys())
            raw_y = [oberkante[px] for px in sorted_x]
            
            # Profil mathematisch glätten
            window_size = max(5, int(w * 0.05))
            conv_kernel = np.ones(window_size) / window_size
            smoothed_y = np.convolve(raw_y, conv_kernel, mode='same')
            
            # 3. Den "Anatomischen Anker" finden (Höchster Punkt = Großer Zeh)
            best_idx = np.argmin(smoothed_y) 
            anchor_x = sorted_x[best_idx]
            
            # Schwerpunkt berechnen, um Links/Rechts Ausrichtung zu prüfen
            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w/2
            
            span = w * 0.65 # Ca. 65% der Fußbreite ist Zehen-Reihe
            
            if anchor_x < center_x:
                # Rechter Fuß (Großer Zeh ist links vom Schwerpunkt)
                start_x = anchor_x - (span * 0.1)
                end_x = anchor_x + span
                fuss_name = "Rechter Fuß"
            else:
                # Linker Fuß (Großer Zeh ist rechts vom Schwerpunkt)
                start_x = anchor_x - span
                end_x = anchor_x + (span * 0.1)
                fuss_name = "Linker Fuß"
                
            # Zonen-Grenzen sichern
            start_x = max(start_x, sorted_x[0])
            end_x = min(end_x, sorted_x[-1])
            
            toe_x = []
            toe_y = []
            for i, sx in enumerate(sorted_x):
                if start_x <= sx <= end_x:
                    toe_x.append(sx)
                    toe_y.append(smoothed_y[i])
                    
            # 4. In 5 Segmente einteilen (für 5 Zehen)
            if len(toe_x) > 10:
                zone_width = toe_x[-1] - toe_x[0]
                segment_width = zone_width / 5.0
                
                toe_count = 1
                for i in range(5):
                    seg_min_x = toe_x[0] + i * segment_width
                    seg_max_x = toe_x[0] + (i + 1) * segment_width
                    
                    segment_indices = [idx for idx, sx in enumerate(toe_x) if seg_min_x <= sx <= seg_max_x]
                    
                    if segment_indices:
                        # Den höchsten Punkt (Minimum Y) im jeweiligen Segment finden
                        local_best_idx = min(segment_indices, key=lambda idx: toe_y[idx])
                        final_px = toe_x[local_best_idx]
                        final_py = oberkante[final_px]
                        
                        # V16: Den Messpunkt ~5% nach unten ins Gelenk schieben (nicht genau auf den Nagel)
                        offset_y = int(h * 0.05) 
                        
                        detected_points.append({
                            "name": f"{fuss_name} - Zeh {toe_count}",
                            "punkt": (int(final_px), int(final_py + offset_y))
                        })
                        toe_count += 1
                        
        return detected_points