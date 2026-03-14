import cv2
from modules.loader import load_and_preprocess
from modules.geometry import analyze_shape_and_draw_skeleton

# Probier "HandLinks.jpeg" oder "FussLinks.jpeg"
img_path = "inputimg/FussLinks.jpeg"
original, gray = load_and_preprocess(img_path)

if original is not None:
    print("--- Thermische Topologie-Analyse gestartet ---")
    
    analyzed_image, method = analyze_shape_and_draw_skeleton(original, gray)
    
    cv2.imshow("Wärmespitzen Scanner", analyzed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()