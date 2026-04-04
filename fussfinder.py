import cv2
import numpy as np
from typing import List, Dict, Any

class FootFinder:
    """
    Kapselt die vollautomatische Erkennung der Zehen (Anatomical Anchor V14).
    Nutzt geometrische Profil-Analyse anstelle von reinen Farbkanälen.
    """

    @staticmethod
    def find_toes(image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analysiert das Wärmebild geometrisch und findet die anatomischen Ankerpunkte (Gelenke).
        """
        # 1. Konvertierung in Graustufen (besser für unterschiedliche Color-Maps)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu-Thresholding um den Fuß sauber vom dunklen Hintergrund zu trennen
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphologische Operationen um Rauschen zu entfernen und Lücken zu schließen
        kernel_small = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((15, 15), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_points = []
        
        if not contours:
            return detected_points
            
        # Finde die zwei größten Konturen (Die beiden Füße)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        # Sortiere Füße von Links nach Rechts im Bild
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
        
        for f_idx, contour in enumerate(sorted_contours):
            if cv2.contourArea(contour) < 5000: # Rauschen ignorieren
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            fuss_name = "Linker Fuß" if f_idx == 0 else "Rechter Fuß"
            
            # Nur das obere Drittel des Fußes betrachten (dort wo die Zehen/Gelenke sind)
            top_limit = y + int(h * 0.40) 
            
            # Profil der Oberkante erstellen: Höchster Punkt (minimales Y) für jedes X
            oberkante = {}
            for pt in contour:
                px, py = pt[0]
                if py < top_limit:
                    if px not in oberkante or py < oberkante[px]:
                        oberkante[px] = py
                        
            if len(oberkante) < 15:
                continue
                
            # X-Koordinaten sortieren für ein fortlaufendes Profil
            sorted_x = sorted(oberkante.keys())
            raw_y = [oberkante[px] for px in sorted_x]
            
            # Profil mathematisch glätten (Moving Average Filter), um kleine Zacken zu ignorieren
            window_size = 15
            if len(raw_y) < window_size:
                continue
                
            kernel = np.ones(window_size) / window_size
            smoothed_y = np.convolve(raw_y, kernel, mode='valid')
            valid_x = sorted_x[window_size//2 : -window_size//2 + 1]
            
            peaks = []
            # Lokale Minima im Profil finden (Y-Achse wächst nach unten, Minima = Zehenspitzen)
            for i in range(3, len(smoothed_y) - 3):
                # Prüfen ob Punkt höher liegt als seine Nachbarn
                if (smoothed_y[i] < smoothed_y[i-1] and smoothed_y[i] < smoothed_y[i-2] and 
                    smoothed_y[i] < smoothed_y[i+1] and smoothed_y[i] < smoothed_y[i+2]):
                    
                    # Prominenz prüfen (Wie tief gehen die Täler daneben?)
                    val_left = max(smoothed_y[max(0, i-15):i]) if i > 0 else smoothed_y[i]
                    val_right = max(smoothed_y[i+1:min(len(smoothed_y), i+16)]) if i < len(smoothed_y)-1 else smoothed_y[i]
                    
                    prominenz = max(val_left - smoothed_y[i], val_right - smoothed_y[i])
                    
                    # Wenn es ein deutlicher Zeh ist
                    if prominenz > 2.0:  
                        px = valid_x[i]
                        py = oberkante[px]
                        
                        # V14 Feature: Nicht die Zehenspitze markieren, sondern das Gelenk darunter!
                        # Wir wandern ~15 Pixel nach unten in die Mitte des Zehs/Gelenks
                        peaks.append((px, py + 15)) 

            # Zu nah beieinander liegende Peaks herausfiltern (Non-Maximum Suppression)
            filtered_peaks = []
            for p in peaks:
                too_close = False
                for fp in filtered_peaks:
                    if abs(p[0] - fp[0]) < 18: # Mindestabstand in X
                        too_close = True
                        break
                if not too_close:
                    filtered_peaks.append(p)
                    
            # Die gefundenen Zehen von Links nach Rechts benennen (Maximal 5)
            for i, (px, py) in enumerate(filtered_peaks[:5]):
                detected_points.append({
                    "name": f"{fuss_name} - Zeh {i+1}",
                    "punkt": (int(px), int(py))
                })
                
        return detected_points