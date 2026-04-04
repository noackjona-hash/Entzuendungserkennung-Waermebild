import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V22).
    Nutzt bewährtes Red-Channel Masking (V13) + High-Pass Prominence Filter gegen Schrägen.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung: Die bewährte V13 Red-Channel Methode
        # Sehr robust für FLIR Ironbow, schneidet kalte Zehen nicht ab.
        _, _, r_channel = cv2.split(image)
        blurred = cv2.GaussianBlur(r_channel, (11, 11), 0)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = int(otsu_val * 0.8)
        _, thresh = cv2.threshold(blurred, final_thresh, 255, cv2.THRESH_BINARY)

        # Morphologische Bereinigung
        k_size = max(5, int(img_w * 0.01))
        kernel = np.ones((k_size, k_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Filtere Noise und wähle die zwei größten Konturen
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
                
            oberkante = {}
            for px, py in top_pts:
                if px not in oberkante or py < oberkante[px]:
                    oberkante[px] = py
                    
            sorted_x = sorted(oberkante.keys())
            raw_y = [oberkante[px] for px in sorted_x]
            
            # 3. High-Pass Filter (Moving Average Baseline)
            # Berechnet die grobe Form des Fußes (Schräge), schneidet aber durch die Zehen.
            window_size = max(15, int(w * 0.25))
            if window_size % 2 == 0: window_size += 1
            pad_size = window_size // 2
            
            padded_y = np.pad(raw_y, (pad_size, pad_size), mode='edge')
            kernel_ma = np.ones(window_size) / window_size
            baseline_y = np.convolve(padded_y, kernel_ma, mode='valid')
            
            # Prominenz: Wie weit sticht der Punkt nach OBEN (kleineres Y) über die Baseline hinaus?
            # Dies neutralisiert jede Schräglage des Fußes vollkommen!
            prominence = [b_y - r_y for b_y, r_y in zip(baseline_y, raw_y)]
            
            # 4. Ankerpunkt (Großer Zeh) finden
            peak_idx = np.argmin(raw_y)
            peak_x = sorted_x[peak_idx]
            center_x = x + w / 2.0
            
            is_right_foot = peak_x < center_x
            
            # Slicing Grenzen definieren
            if is_right_foot:
                start_x = peak_x - int(w * 0.10)
                end_x = peak_x + int(w * 0.75) 
            else:
                start_x = peak_x - int(w * 0.75) 
                end_x = peak_x + int(w * 0.10)
                
            start_x = max(start_x, sorted_x[0])
            end_x = min(end_x, sorted_x[-1])
            
            # 5. In 5 exakte Spalten aufteilen und Prominenz-Peaks suchen
            span_width = end_x - start_x
            if span_width <= 0: continue
            segment_width = span_width / 5.0
            foot_points = []
            
            for i in range(5):
                seg_start = start_x + i * segment_width
                seg_end = start_x + (i + 1) * segment_width
                
                # Indizes der Pixel in diesem Segment
                seg_indices = [j for j, px in enumerate(sorted_x) if seg_start <= px <= seg_end]
                
                if seg_indices:
                    # DEN PUNKT MIT DER HÖCHSTEN PROMINENZ FINDEN (Der wahre Zeh!)
                    best_idx = max(seg_indices, key=lambda j: prominence[j])
                    best_px = sorted_x[best_idx]
                    best_py = raw_y[best_idx]
                    
                    # Punkt ca. 4% der Fußhöhe nach unten ins Gelenk verschieben
                    offset_y = max(5, int(h * 0.04))
                    foot_points.append({"x": best_px, "y": best_py + offset_y})
                else:
                    # Fallback für eine komplett leere Spalte
                    fallback_x = int(seg_start + segment_width / 2)
                    closest_idx = np.argmin([abs(sx - fallback_x) for sx in sorted_x])
                    fallback_y = raw_y[closest_idx]
                    foot_points.append({"x": fallback_x, "y": fallback_y + int(h * 0.04)})
                    
            # 6. Anatomisch korrekte Benennung
            for i, pt in enumerate(foot_points):
                zeh_nr = i + 1 if is_right_foot else 5 - i
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y']))
                })
                
        return detected_points