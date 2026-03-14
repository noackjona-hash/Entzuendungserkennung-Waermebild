import cv2

def apply_smart_heat_mask(gray_img, sensitivity=15):
    """
    Erstellt eine Maske basierend auf dem heißesten Punkt im Bild.
    sensitivity: Wie weit der Wert vom Maximum abweichen darf, um noch markiert zu werden.
    """
    # 1. Den absoluten Hitzepunkt (Helligkeits-Maximum) finden
    _, max_val, _, _ = cv2.minMaxLoc(gray_img)
    
    # 2. Dynamischen Schwellenwert berechnen (Maximum minus Empfindlichkeit)
    # So isolieren wir wirklich nur die absolute Spitze der Hitzequelle
    dynamic_threshold = max_val - sensitivity
    
    # 3. Maske anwenden
    _, mask = cv2.threshold(gray_img, dynamic_threshold, 255, cv2.THRESH_BINARY)
    
    # 4. Kleine Bildfehler/Rauschen entfernen
    mask = cv2.medianBlur(mask, 5)
    
    return mask, max_val