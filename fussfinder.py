import cv2
import numpy as np
from scipy.signal import find_peaks

class FootFinder:
    """
    Anatomische Orientierung V18: Nutzt Red-Channel-Extraction anstatt Graustufen,
    um FLIR-Farbpaletten (Ironbow) perfekt vom Hintergrund zu isolieren.
    """
    @staticmethod
    def find_toes(image: np.ndarray) -> list:
        h, w = image.shape[:2]
        
        # PROFI-TRICK: Bei Wärmebildern (Ironbow) ist der rote Kanal (Index 2 in BGR) 
        # der absolut stärkste Indikator für Hitze (Weiß, Gelb, Orange, Rot).
        # Blau und Violett (Hintergrund) verschwinden fast komplett.
        red_channel = image[:, :, 2]
        
        # Weichzeichnen, um Rauschen zu killen
        blurred = cv2.GaussianBlur(red_channel, (11, 11), 0)
        
        # Wir nehmen nur Pixel, die mindestens 50% der Maximalhitze des Bildes haben
        _, max_val, _, _ = cv2.minMaxLoc(blurred)
        thresh_val = max_val * 0.5 
        
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Löcher in den Füßen stopfen
        kernel = np.ones((11,11), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ein echter Fuß muss riesig sein (mind. 5% des Bildes)
        valid_feet = [c for c in contours if cv2.contourArea(c) > (w * h * 0.05)]
        
        # Wenn wir nicht exakt 2 Füße sehen -> Abbruch -> Fallback auf UniversalFinder
        if len(valid_feet) != 2:
            return []
            
        valid_feet = sorted(valid_feet, key=lambda c: cv2.boundingRect(c)[0])
        detected_points = []
        
        for i, foot in enumerate(valid_feet):
            fx, fy, fw, fh = cv2.boundingRect(foot)
            side = "Linker Fuß" if i == 0 else "Rechter Fuß"
            
            # Profil der Oberkante des Fußes bauen
            top_edge = np.full(fw, fh, dtype=int)
            for pt in foot:
                px, py = pt[0]
                rel_x = px - fx
                rel_y = py - fy
                if rel_y < top_edge[rel_x]:
                    top_edge[rel_x] = rel_y
                    
            # Profil glätten und umdrehen für den Peak-Finder
            smoothed = np.convolve(top_edge, np.ones(15)/15, mode='same')
            inverted = fh - smoothed
            
            # Zehen-Spitzen (Peaks) suchen
            peaks, _ = find_peaks(inverted, distance=max(10, fw//7), prominence=5)
            peaks = sorted(peaks)
            
            # Wenn es total wild aussieht (keine Zehen oder zu viele) -> Ignorieren
            if not (1 <= len(peaks) <= 6):
                continue 
                
            for j, p in enumerate(peaks[:5]): 
                px = fx + p
                py = fy + int(top_edge[p]) + 15 
                
                # Anatomisch korrekt nummerieren
                num = j + 1 if i == 1 else len(peaks[:5]) - j
                detected_points.append({"name": f"{side} - Zeh {num}", "punkt": (px, py)})
                
        return detected_points