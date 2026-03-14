import cv2

def get_extremities_roi(gray_img):
    """Sucht die Hand oder den Fuß und gibt den oberen Teil (Finger/Zehen) als ROI zurück."""
    # 1. Objekt vom Hintergrund trennen
    _, body_mask = cv2.threshold(gray_img, 40, 255, cv2.THRESH_BINARY)
    
    # 2. Umrisse finden
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # 3. Den größten Umriss nehmen (die Hand oder den Fuß)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # 4. Suchzone berechnen: Finger sind etwas länger als Zehen, 
    # daher nehmen wir die oberen 30% des erkannten Objekts (statt 25%).
    extremity_h = int(h * 0.30)
    
    return (x, y, w, extremity_h)