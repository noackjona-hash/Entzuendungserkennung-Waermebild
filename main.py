import cv2
from modules.loader import load_and_preprocess
from modules.masking import apply_smart_heat_mask
from modules.analysis import get_hotspot_bounding_box
from modules.anatomy import get_extremities_roi # Neuer, universeller Name!

# Teste hier jetzt auch mal "inputimg/HandLinks.jpeg" oder "inputimg/HandRechts.jpeg"
img_path = "inputimg/FussLinks.jpeg" 
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Automatische Extremitaeten-Analyse gestartet ---")
    
    # 1. Finger/Zehenbereich automatisch finden
    roi = get_extremities_roi(gray)
    
    if roi:
        x, y, w, h = roi
        
        # Blaue Box für die Suchzone
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(original, "Suchzone (Finger/Zehen)", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Bildausschnitt für die Analyse
        roi_gray = gray[y:y+h, x:x+w]
        
        # Smarte Maske NUR in der Suchzone anwenden
        mask, max_val = apply_smart_heat_mask(roi_gray, sensitivity=15)
        
        # Bounding Box finden
        bbox = get_hotspot_bounding_box(mask)
        
        if bbox:
            bx, by, bw, bh = bbox
            # Grüne Box für den Hotspot
            cv2.rectangle(original, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 255, 0), 2)
            cv2.putText(original, f"Hotspot: {max_val}", (x + bx, y + by - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Hotspot in Extremitaet gefunden! Intensität: {max_val}")
        else:
            print("Kein signifikanter Hotspot in der Suchzone gefunden.")
    else:
        print("Fehler: Konnte weder Hand noch Fuss im Bild erkennen.")

    # Endergebnis anzeigen
    cv2.imshow("Automatische Diagnose", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()