import cv2
from modules.loader import load_and_preprocess
from modules.masking import apply_smart_heat_mask
from modules.analysis import get_hotspot_bounding_box

img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Medizinisches Analyse-Tool ---")
    print("1. Markiere mit der Maus das zu untersuchende Gelenk (z.B. den grossen Zeh).")
    print("2. Druecke danach ENTER oder die LEERTASTE.")
    
    # Nutzer wählt einen Bereich (Region of Interest = ROI) aus
    roi = cv2.selectROI("Gelenk markieren", original, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Gelenk markieren")
    
    x, y, w, h = roi
    
    # Nur weitermachen, wenn auch wirklich ein Bereich markiert wurde
    if w > 0 and h > 0:
        # Wir schneiden das Bild auf den markierten Bereich zu
        roi_gray = gray[y:y+h, x:x+w]
        
        # Smarte Maske NUR auf das markierte Gelenk anwenden
        # Da ein Zeh generell kälter ist, hilft diese lokale Analyse extrem!
        mask, max_val = apply_smart_heat_mask(roi_gray, sensitivity=15)
        
        # Bounding Box im kleinen Ausschnitt finden
        bbox = get_hotspot_bounding_box(mask)
        
        if bbox:
            bx, by, bw, bh = bbox
            # Rechteck an die richtige Stelle im Originalbild zeichnen
            cv2.rectangle(original, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 255, 0), 2)
            cv2.putText(original, "Gelenk-Entzuendung", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Hotspot im Gelenk gefunden! Lokaler Spitzenwert: {max_val}")

    # Endergebnis anzeigen
    cv2.imshow("Gelenk-Analyse", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()