import cv2
from modules.loader import load_and_preprocess
from modules.geometry import analyze_extremities_and_smart_heat # <-- HIER IST DER NEUE NAME

# Probier hier dein Fuß- oder Handbild
img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Smarter Entzuendungs-Scanner gestartet ---")
    
    # HIER AUCH DEN NEUEN NAMEN NUTZEN
    analyzed_image = analyze_extremities_and_smart_heat(original, gray)
    
    cv2.imshow("Smarte Diagnose", analyzed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()