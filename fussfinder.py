import cv2
import numpy as np

class FootFinder:
    """Anatomische Orientierung V24: Sucht Füße und Zehenstrukturen."""
    @staticmethod
    def find_toes(image: np.ndarray) -> list:
        h, w = image.shape[:2]
        _, _, r = cv2.split(image)
        
        # Rauschen entfernen und Binarisieren
        blurred = cv2.GaussianBlur(r, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphologisches Schließen für solide Flächen
        kernel = np.ones((7,7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_feet = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        valid_feet = sorted(valid_feet, key=lambda c: cv2.boundingRect(c)[0])[:2]
        
        points = []
        for i, foot in enumerate(valid_feet):
            x, y, fw, fh = cv2.boundingRect(foot)
            side = "Linker Fuß" if i == 0 else "Rechter Fuß"
            
            # 5 Segmente pro Fuß (5 Zehen)
            for step in range(5):
                # Erzeuge Maske für das Segment (Vertikaler Slice im oberen Bereich)
                mask = np.zeros(thresh.shape, dtype="uint8")
                segment_x_start = x + int(step * (fw/5))
                segment_x_end = x + int((step+1) * (fw/5))
                cv2.rectangle(mask, (segment_x_start, y), (segment_x_end, y + int(fh*0.4)), 255, -1)
                
                intersect = cv2.bitwise_and(thresh, mask)
                M = cv2.moments(intersect)
                
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    # Anatomisch korrekt: Linker Fuß = großer Zeh rechts (Step 4), Rechter Fuß = großer Zeh links (Step 0)
                    is_right = (i == 1)
                    num = step + 1 if is_right else 5 - step
                    points.append({"name": f"{side} - Zeh {num}", "punkt": (cx, cy + int(fh*0.05))})
                    
        return pointss