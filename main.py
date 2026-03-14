import cv2
from modules.loader import load_and_preprocess
from modules.masking import apply_smart_heat_mask
from modules.analysis import get_hotspot_bounding_box

img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    # 1. Smarte Maske erstellen
    mask, max_val = apply_smart_heat_mask(gray, sensitivity=10)
    
    # 2. Bounding Box finden
    bbox = get_hotspot_bounding_box(mask)
    
    if bbox:
        x, y, w, h = bbox
        # Grünes Rechteck ins Originalbild zeichnen
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(original, "Hotspot erkannt", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Ergebnisse anzeigen
    cv2.imshow("KI Erkennung", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()