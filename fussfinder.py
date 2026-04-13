import cv2
import numpy as np
from scipy.signal import find_peaks

class FootFinder:
    """
    Anatomische Orientierung V16: Nutzt Topologische Peak-Detection (Signalverarbeitung),
    um die exakten Zehenspitzen auf der Wärmekontur zu finden.
    Besser und deterministischer als Standard-KI-Modelle.
    """
    @staticmethod
    def find_toes(image: np.ndarray) -> list:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE für besseren Kontrast bei FLIR-Farbpaletten
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Adaptive Binarisierung
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Rauschen und kleine Lücken entfernen
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Finde die zwei größten Wärmequellen (Füße/Beine)
        valid_feet = sorted([c for c in contours if cv2.contourArea(c) > (w * h * 0.02)], key=cv2.contourArea, reverse=True)[:2]
        valid_feet = sorted(valid_feet, key=lambda c: cv2.boundingRect(c)[0]) # Von Links nach Rechts sortieren
        
        detected_points = []
        
        for i, foot in enumerate(valid_feet):
            fx, fy, fw, fh = cv2.boundingRect(foot)
            side = "Linker Fuß" if i == 0 else "Rechter Fuß"
            
            # 1. Erstelle ein Profil der Oberkante des Fußes
            top_edge = np.full(fw, fh, dtype=int)
            for pt in foot:
                px, py = pt[0]
                rel_x = px - fx
                rel_y = py - fy
                if rel_y < top_edge[rel_x]:
                    top_edge[rel_x] = rel_y
                    
            # 2. Glätten des Kanten-Profils
            smoothed = np.convolve(top_edge, np.ones(10)/10, mode='same')
            
            # 3. Peak-Finding (Täler in der Y-Achse sind Spitzen im Bild)
            inverted = fh - smoothed
            
            # Finde die Zehen-Spitzen (Peaks)
            peaks, _ = find_peaks(inverted, distance=max(10, fw//8), prominence=3)
            
            peaks = sorted(peaks)
            
            for j, p in enumerate(peaks[:5]): # Maximal 5 Zehen
                px = fx + p
                py = fy + int(top_edge[p]) + 15 # 15 Pixel nach unten, um im Zentrum der Zehe zu sein
                
                # Anatomisch korrekt benennen
                num = j + 1 if i == 1 else 5 - j
                detected_points.append({"name": f"{side} - Zeh {num}", "punkt": (px, py)})
                
        return detected_points