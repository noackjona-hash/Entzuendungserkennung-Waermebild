import cv2

def apply_heat_mask(gray_img, threshold_value=200):
    # Erstellt eine Maske: Alles über dem Wert wird weiß (255), der Rest schwarz (0)
    _, mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Optional: Rauschen entfernen (Morphologische Operationen)
    mask = cv2.medianBlur(mask, 5)
    
    return mask