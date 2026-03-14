# Erst die Importe (ganz oben)
from modules.loader import load_and_preprocess
from modules.masking import apply_heat_mask
import cv2

# Dann die Logik
img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    heat_mask = apply_heat_mask(gray, threshold_value=210)

    cv2.imshow("Original", original)
    cv2.imshow("Maske", heat_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows() # Dieser Befehl steht ganz am Ende