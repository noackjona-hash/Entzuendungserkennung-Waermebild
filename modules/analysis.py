import cv2
import numpy as np

def calculate_hotspot_area(mask):
    # Zählt alle Pixel, die nicht schwarz (0) sind
    white_pixels = cv2.countNonZero(mask)
    
    # Gesamtanzahl der Pixel im Bild
    total_pixels = mask.shape[0] * mask.shape[1]
    
    # Prozentsatz berechnen
    percentage = (white_pixels / total_pixels) * 100
    
    return white_pixels, percentage