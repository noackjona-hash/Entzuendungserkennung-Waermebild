import cv2
from modules.loader import load_and_preprocess
from modules.geometry import analyze_shape_and_draw_skeleton

# Probier hier "HandLinks.jpeg" oder "FussLinks.jpeg"
img_path = "inputimg/HandLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Algorithmische Geometrie-Analyse gestartet ---")
    
    # Reine Mathematik auf das Bild anwenden
    analyzed_image, part_type = analyze_shape_and_draw_skeleton(original, gray)
    
    # Ergebnis auf das Bild schreiben
    if part_type == "Hand":
        print("Klassifizierung: Hand")
        cv2.putText(analyzed_image, "Algorithmus erkennt: Hand", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        print("Klassifizierung: Fuss")
        cv2.putText(analyzed_image, "Algorithmus erkennt: Fuss", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Geometrisches Skelett", analyzed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()