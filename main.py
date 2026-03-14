from modules.loader import load_and_preprocess
from modules.masking import apply_heat_mask
import cv2

# Pfad zum Testbild
img_path = "inputimg/FussLinks.jpeg"

# SCHRITT 1: Laden
original, gray = load_and_preprocess(img_path)

if original is not None:
    # SCHRITT 2: Maskieren (Wir suchen nur die heißesten Stellen)
    # Spiel hier mit dem Wert (z.B. 220 statt 200), um die Maske zu verfeinern
    heat_mask = apply_heat_mask(gray, threshold_value=210)

    # Ergebnis anzeigen
    cv2.imshow("Original", original)
    cv2.imshow("Nur Hotspots (Maske)", heat_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()from modules.loader import load_and_preprocess
from modules.masking import apply_heat_mask
import cv2

# Pfad zum Testbild
img_path = "inputimg/FussLinks.jpeg"

# SCHRITT 1: Laden
original, gray = load_and_preprocess(img_path)

if original is not None:
    # SCHRITT 2: Maskieren (Wir suchen nur die heißesten Stellen)
    # Spiel hier mit dem Wert (z.B. 220 statt 200), um die Maske zu verfeinern
    heat_mask = apply_heat_mask(gray, threshold_value=210)

    # Ergebnis anzeigen
    cv2.imshow("Original", original)
    cv2.imshow("Nur Hotspots (Maske)", heat_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()