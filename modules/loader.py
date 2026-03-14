import cv2
import numpy as np
import os

def load_and_preprocess(image_path):
    if not os.path.exists(image_path):
        print(f"FEHLER: Das Bild '{image_path}' existiert nicht.")
        return None, None
        
    # BUGFIX: Umlaute im Pfad umgehen (cv2.imread scheitert an 'ä', 'ü' etc.)
    # Wir lesen die Datei als rohen Byte-Stream mit NumPy
    file_bytes = np.fromfile(image_path, np.uint8)
    
    # ... und lassen OpenCV das Bild aus dem Speicher decodieren
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"FEHLER: Konnte das Bild nicht decodieren: {image_path}")
        return None, None

    # Bild etwas verkleinern, damit es gut in die GUI passt (max Hoehe 600px)
    h, w = img.shape[:2]
    if h > 600:
        ratio = 600 / h
        img = cv2.resize(img, (int(w * ratio), 600))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray