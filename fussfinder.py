import cv2
import numpy as np
import math

class FootFinder:
    """Anatomical Anchor V24: Detektiert Zehenstrukturen in thermischen Clustern."""
    @staticmethod
    def find_toes(image: np.ndarray) -> list:
        h, w = image.shape[:2]
        # Red-Channel Analyse für thermische Signaturen
        _, _, r = cv2.split(image)
        blurred = cv2.GaussianBlur(r, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_feet = [c for c in contours if cv2.contourArea(c) > (w * h * 0.02)]
        valid_feet = sorted(valid_feet, key=lambda c: cv2.boundingRect(c)[0])[:2]
        
        points = []
        for i, foot in enumerate(valid_feet):
            x, y, fw, fh = cv2.boundingRect(foot)
            side = "Linker Fuß" if i == 0 else "Rechter Fuß"
            
            # Analyse der Oberkante (Zehenregion)
            for step in range(5):
                segment_x = x + int((step + 0.5) * (fw / 5))
                # Finde den höchsten Punkt in diesem Segment
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.rectangle(mask, (x + int(step * (fw/5)), y), (x + int((step+1)*(fw/5)), y + int(fh*0.4)), 255, -1)
                intersect = cv2.bitwise_and(thresh, mask)
                
                M = cv2.moments(intersect)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    # Anatomische Korrektur: Großer Zeh ist bei Füßen innen
                    is_right = (i == 1)
                    num = step + 1 if is_right else 5 - step
                    points.append({"name": f"{side} - Zeh {num}", "punkt": (cx, cy + 10)})
        return points