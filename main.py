from modules.loader import load_and_preprocess
from modules.masking import apply_heat_mask
from modules.analysis import calculate_hotspot_area # NEU
import cv2

img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    # Maske erstellen
    heat_mask = apply_heat_mask(gray, threshold_value=210)

    # NEU: Analyse ausführen
    pixel_count, percent = calculate_hotspot_area(heat_mask)
    
    print(f"Analyse-Ergebnis für {img_path}:")
    print(f"- Heisse Pixel: {pixel_count}")
    print(f"- Anteil am Gesamtbild: {percent:.2f}%")

    # Ergebnisse anzeigen
    cv2.imshow("Original", original)
    cv2.imshow("Maske", heat_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()