import cv2
import numpy as np
from scipy.signal import find_peaks

class FootFinder:
    """
    Anatomische Orientierung V17: Nutzt weiche Schwellenwerte, um
    Wärmebild-Farbpaletten besser zu segmentieren, inklusive strikter 
    Validierung für den Universal-Fallback.
    """
    @staticmethod
    def find_toes(image: np.ndarray) -> list:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Weicher Blur, um Bildrauschen im Hintergrund zu glätten
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Dynamischer Threshold: Schneidet den tiefschwarzen/lilanen Hintergrund ab,
        # behält aber die kühleren (roten) Ränder des Fußes.
        mean_val = np.mean(blurred)
        thresh_val = max(25, mean_val * 0.6)
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Morphologische Operationen, um Risse im Fuß zu schließen
        kernel = np.ones((9,9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Strikte Filterung: Ein echter Fuß muss auf diesem Bild groß sein (mind. 4% Fläche)
        valid_feet = [c for c in contours if cv2.contourArea(c) > (w * h * 0.04)]
        
        # VALIDIERUNGS-CHECK: Wenn wir nicht exakt 2 Füße haben, ist das Bild
        # zu unsauber. Wir brechen ab, damit der UniversalFinder übernimmt!
        if len(valid_feet) != 2:
            return []
            
        valid_feet = sorted(valid_feet, key=lambda c: cv2.boundingRect(c)[0])
        detected_points = []
        
        for i, foot in enumerate(valid_feet):
            fx, fy, fw, fh = cv2.boundingRect(foot)
            side = "Linker Fuß" if i == 0 else "Rechter Fuß"
            
            # Sicherheits-Check: Ein Fuß ist typischerweise höher als breit
            if fw > fh * 1.5:
                return []
                
            top_edge = np.full(fw, fh, dtype=int)
            for pt in foot:
                px, py = pt[0]
                rel_x = px - fx
                rel_y = py - fy
                if rel_y < top_edge[rel_x]:
                    top_edge[rel_x] = rel_y
                    
            smoothed = np.convolve(top_edge, np.ones(15)/15, mode='same')
            inverted = fh - smoothed
            
            # Höhere Prominenz, um "falsche" kleine Hubbel am Fuß zu ignorieren
            peaks, _ = find_peaks(inverted, distance=max(10, fw//7), prominence=5)
            peaks = sorted(peaks)
            
            # Sicherheits-Check: Wir erwarten etwa 5 Zehen. Bei kompletter Abweichung -> Fallback
            if not (2 <= len(peaks) <= 6):
                return []
                
            for j, p in enumerate(peaks[:5]): 
                px = fx + p
                py = fy + int(top_edge[p]) + 15 
                num = j + 1 if i == 1 else len(peaks[:5]) - j
                detected_points.append({"name": f"{side} - Zeh {num}", "punkt": (px, py)})
                
        return detected_points