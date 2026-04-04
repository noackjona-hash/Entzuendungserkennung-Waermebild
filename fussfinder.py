import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V23).
    Nutzt 'Dynamic Poly-Boundary Fusion' & 'Local Maxima Scanning'.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        
        # 1. Maskierung: Red-Channel + Dynamic Otsu Fusion
        _, _, r_channel = cv2.split(image)
        blurred = cv2.GaussianBlur(r_channel, (11, 11), 0)
        
        # Dynamischer Threshold für robuste Erkennung auch bei kälteren Füßen
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = max(15, int(otsu_val * 0.75)) 
        _, thresh = cv2.threshold(blurred, final_thresh, 255, cv2.THRESH_BINARY)

        # Morphologie zur Glättung
        k_size = max(5, int(img_w * 0.01))
        kernel = np.ones((k_size, k_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours: 
            return detected_points
            
        # Filtere Noise und hole die zwei größten Konturen (Füße)
        valid_contours = [c for c in contours if cv2.contourArea(c) > (img_w * img_h * 0.01)]
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0]) 
        
        for f_idx, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # 2. Oberkante extrahieren (oberste 30%)
            top_limit = y + int(h * 0.30) 
            top_pts = [pt[0] for pt in contour if pt[0][1] < top_limit]
            
            if len(top_pts) < (w * 0.2): 
                continue
                
            # Dictionary für die höchste Y-Koordinate pro X-Pixel
            oberkante = {}
            for px, py in top_pts:
                if px not in oberkante or py < oberkante[px]:
                    oberkante[px] = py
                    
            sorted_x = sorted(oberkante.keys())
            raw_y = [oberkante[px] for px in sorted_x]
            
            # 3. Mathematische Glättung der Kontur (Noise Reduction)
            # Wichtig: Kernel size darf nicht zu groß sein, sonst verschmelzen Zehen
            window_size = max(5, int(w * 0.03)) 
            if window_size % 2 == 0: window_size += 1
            
            padded_y = np.pad(raw_y, (window_size//2, window_size//2), mode='edge')
            kernel_smooth = np.ones(window_size) / window_size
            smoothed_y = np.convolve(padded_y, kernel_smooth, mode='valid')

            # 4. Local Maxima Scanning (Suche nach echten "Gipfeln" der Zehen)
            # Y wächst nach unten, also suchen wir nach lokalen Minima in smoothed_y
            peaks = []
            for i in range(2, len(smoothed_y) - 2):
                if (smoothed_y[i] < smoothed_y[i-1] and smoothed_y[i] < smoothed_y[i-2] and
                    smoothed_y[i] < smoothed_y[i+1] and smoothed_y[i] < smoothed_y[i+2]):
                    peaks.append(i)

            foot_points = []
            
            # 5. Peak Filtering & Fallbacks
            if len(peaks) > 0:
                # Extrahiere X/Y für die gefundenen Peaks
                peak_coords = [(sorted_x[idx], raw_y[idx]) for idx in peaks]
                
                # Filtere Peaks, die zu nah aneinander liegen (mind. 5% Fußbreite Abstand)
                min_dist = w * 0.05
                filtered_peaks = []
                for pt in sorted(peak_coords, key=lambda p: p[1]): # Sortiere nach Y (höchste zuerst)
                    too_close = False
                    for fp in filtered_peaks:
                        if abs(pt[0] - fp[0]) < min_dist:
                            too_close = True
                            break
                    if not too_close:
                        filtered_peaks.append(pt)
                
                # Wir wollen max 5 Zehen
                filtered_peaks = sorted(filtered_peaks, key=lambda p: p[0])[:5] # Von links nach rechts
                
                # Wenn wir weniger als 5 Zehen gefunden haben, füllen wir interpoliert auf
                if len(filtered_peaks) < 5:
                    # Ankerpunkt suchen (absolutes Minimum Y)
                    peak_x = min(filtered_peaks, key=lambda p: p[1])[0]
                    center_x = x + w / 2.0
                    is_right_foot = peak_x < center_x
                    
                    # Definiere Suchbereich
                    if is_right_foot:
                        start_x, end_x = peak_x - int(w * 0.1), peak_x + int(w * 0.7)
                    else:
                        start_x, end_x = peak_x - int(w * 0.7), peak_x + int(w * 0.1)
                    
                    start_x, end_x = max(start_x, sorted_x[0]), min(end_x, sorted_x[-1])
                    
                    # Generiere 5 gleichmäßige Spalten als Fallback
                    span = end_x - start_x
                    if span > 0:
                        seg_w = span / 5.0
                        foot_points = []
                        for i in range(5):
                            target_x = start_x + i * seg_w + (seg_w / 2)
                            # Finde den am nächsten liegenden gefilterten Peak
                            closest_peak = None
                            min_diff = float('inf')
                            for fp in filtered_peaks:
                                diff = abs(fp[0] - target_x)
                                if diff < min_diff and diff < (seg_w * 1.5): # Darf nicht ewig weit weg sein
                                    closest_peak = fp
                                    min_diff = diff
                            
                            if closest_peak:
                                foot_points.append({"x": closest_peak[0], "y": closest_peak[1]})
                            else:
                                # Fallback: Nimm den Punkt direkt auf der Kontur
                                closest_x_idx = np.argmin([abs(sx - target_x) for sx in sorted_x])
                                foot_points.append({"x": sorted_x[closest_x_idx], "y": raw_y[closest_x_idx]})
                else:
                    for p in filtered_peaks:
                        foot_points.append({"x": p[0], "y": p[1]})
            else:
                 # Fallback, falls GAR KEINE Peaks gefunden wurden (sehr unwahrscheinlich bei V23)
                 center_x = x + w / 2.0
                 foot_points = [{"x": int(center_x), "y": int(y + h * 0.1)}] * 5

            # 6. Anatomisch korrekte Benennung & Offset
            # Wir müssen bestimmen, ob der Große Zeh links oder rechts liegt, um korrekt durchzuzählen
            absolute_min_y = min([p["y"] for p in foot_points])
            peak_x = [p["x"] for p in foot_points if p["y"] == absolute_min_y][0]
            is_right_foot = peak_x < (x + w / 2.0)

            for i, pt in enumerate(foot_points):
                # Offset: Punkt minimal nach unten ins Gelenk verschieben
                offset_y = max(5, int(h * 0.04))
                
                # Zuweisung: Läuft immer von Links nach Rechts
                zeh_nr = i + 1 if is_right_foot else 5 - i
                
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {zeh_nr}",
                    "punkt": (int(pt['x']), int(pt['y'] + offset_y))
                })
                
        return detected_points