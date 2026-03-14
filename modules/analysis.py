import cv2

def get_hotspot_bounding_box(mask):
    """Findet die Konturen in der Maske und gibt das größte Rechteck zurück."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Die größte Kontur finden (vermeidet Markierung von kleinem Rauschen)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

def calculate_hotspot_area(mask):
    white_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    percentage = (white_pixels / total_pixels) * 100
    return white_pixels, percentage