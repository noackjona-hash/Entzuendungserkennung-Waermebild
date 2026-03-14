import cv2

def get_toes_roi(gray_img):
    """Sucht den gesamten Fuß und gibt das obere Viertel (die Zehen) als ROI zurück."""
    # 1. Fuß vom Hintergrund trennen (alles, was wärmer als der sehr kalte Hintergrund ist)
    _, foot_mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    
    # 2. Umrisse (Konturen) finden
    contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # 3. Den größten Umriss nehmen (das ist garantiert der Fuß, nicht ein Staubkorn)
    foot_contour = max(contours, key=cv2.contourArea)
    fx, fy, fw, fh = cv2.boundingRect(foot_contour)
    
    # 4. Das obere Viertel berechnen (25% der Gesamthöhe des Fußes)
    toe_h = int(fh * 0.25)
    
    # Koordinaten (x, y, breite, höhe) der Zehen-Region zurückgeben
    return (fx, fy, fw, toe_h)