import cv2
import numpy as np

class UniversalFinder:
    """
    Anatomie-unabhängige Erkennung von Hitzezentren (Hotspots).
    Wird als Fallback genutzt, wenn keine spezifischen Körperteile (wie Füße) erkannt werden.
    """
    @staticmethod
    def find_hotspots(image: np.ndarray, max_regions: int = 6) -> list:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Starker Weichzeichner, um Rauschen zu ignorieren und Kern-Zentren zu finden
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Finde den absolut wärmsten Punkt im Bild
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Wenn das Bild komplett kalt/schwarz ist, brich ab
        if max_val < 50:
            return []
            
        # Binarisierung: Isoliere alles, was mindestens 80% der Maximaltemperatur hat
        threshold_value = max_val * 0.8
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Konturen der Hotspots finden
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Nach Größe (Fläche) sortieren, um winzige Störpixel zu filtern
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        detected_points = []
        for i, cnt in enumerate(contours[:max_regions]):
            # Mindestgröße für einen gültigen Hotspot
            if cv2.contourArea(cnt) < (w * h * 0.005):
                continue
                
            # Berechne den geometrischen Schwerpunkt (Zentrum) der Hitzequelle
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Fallback auf Bounding-Box Mitte
                x, y, bw, bh = cv2.boundingRect(cnt)
                cX = x + bw // 2
                cY = y + bh // 2
                
            detected_points.append({
                "name": f"Universal Region {i+1}", 
                "punkt": (cX, cY)
            })
            
        return detected_points