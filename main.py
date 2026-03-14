import cv2
from modules.loader import load_and_preprocess
from modules.masking import apply_smart_heat_mask
from modules.analysis import get_hotspot_bounding_box
from modules.anatomy import get_toes_roi # Unser neues Modul!

img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Automatische Zehen-Analyse gestartet ---")
    
    # 1. Zehenbereich automatisch finden
    roi = get_toes_roi(gray)
    
    if roi:
        x, y, w, h = roi
        
        # Zur Visualisierung: Zeichne ein blaues Rechteck um die Suchzone (Zehenbereich)
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(original, "Suchzone (Zehen)", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 2. Bildausschnitt für die Analyse erstellen
        roi_gray = gray[y:y+h, x:x+w]
        
        # 3. Smarte Maske NUR im Zehenbereich anwenden
        mask, max_val = apply_smart_heat_mask(roi_gray, sensitivity=15)
        
        # 4. Bounding Box für den Entzündungs-Hotspot finden
        bbox = get_hotspot_bounding_box(mask)
        
        if bbox:
            bx, by, bw, bh = bbox
            # Grünes Rechteck an die richtige Stelle im Originalbild zeichnen
            cv2.rectangle(original, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 255, 0), 2)
            cv2.putText(original, f"Hotspot: {max_val}", (x + bx, y + by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Hotspot automatisch gefunden! Intensität: {max_val}")
        else:
            print("Kein signifikanter Hotspot in der Suchzone gefunden.")
    else:
        print("Fehler: Konnte den Fuß im Bild nicht erkennen.")

    # Endergebnis anzeigen
    cv2.imshow("Automatische Diagnose", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()